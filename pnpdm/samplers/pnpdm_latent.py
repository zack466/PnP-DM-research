import torch, os
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from collections import defaultdict
from torchvision.utils import save_image
from PIL import Image

def norm_image_01(x):
    return (x * 0.5 + 0.5).clip(0, 1)


def visualize_pil(target, figsize=None):
    pil_image = Image.open(target)
    if figsize is None:
        plt.figure(figsize=(24, 24))
    else:
        plt.figure(figsize=figsize)
    plt.imshow(pil_image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()


def visualize_grid(images, target='image.png', nrow=10, figsize=None, normalize=True):
    # Save images.
    save_grid(images, target, nrow=nrow, normalize=normalize)
    visualize_pil(target, figsize=figsize)

def save_grid(images, target='image.png', nrow=10, normalize=True):
    # Save images.
    if normalize:
        images = norm_image_01(images)
    save_image(images, target, nrow=nrow)

class PnPDMLatent:
    """
    Run our latent version of PnP-DM. This is very similar to the original
    versions with modifications to the likelihood and prior steps which allow
    the algorithm to run in latent space.
    """
    def __init__(self, config, model, operator, noiser, device):
        self.config = config
        self.model = model
        self.operator = operator
        self.noiser = noiser
        self.device = device

        self.rho_initial = config.rho_initial
        self.rho_final = config.rho_final

        self.model = model
        assert hasattr(self.model, "num_steps"), "this sampler requires the model to have a `num_steps` attribute"
        assert hasattr(self.model, "decode_image"), "this sampler requires the model to have a `decode_image` function"
        if not hasattr(self.model, "set_prompt"):
            logging.warning("the model for this sampler has no `set_prompt` function")

    @property
    def display_name(self):
        return f'pnpdm-likelihood-{self.model.__class__.__name__}-prior'

    def force(self, x_cur, x_initial, y, sigma, rho):
        # forward operator is A(D(z))
        x_cur2 = x_cur.clone()
        x_cur2.requires_grad = True
        decoded = self.model.decode_image(x_cur2).to(torch.float32)  # super-resolution only supports float32
        data_fit = (self.operator.forward(decoded).to(torch.float16) - y).norm()**2 / (2*sigma**2)
        grad = torch.autograd.grad(outputs=data_fit, inputs=x_cur2)[0]
        return -(grad + (x_cur - x_initial)/rho**2)

    def mcmc_sample(self, _, x0, measurement, sigma, rho):
        lr = 1e-4
        num_steps = 30
        momentum = 0.45

        velocity = torch.randn_like(x0)

        x = x0.clone().detach()
        pbar = trange(num_steps, disable=True)

        for _ in pbar:
            # Langevin step: compute/approximate the score function p(x_0 = x | x_t, y)
            cur_score = self.force(x, x0, measurement, sigma, rho)
            epsilon = torch.randn_like(x)

            # update
            step_size = np.sqrt(lr)
            velocity = momentum * velocity + step_size * cur_score + np.sqrt(2 * (1 - momentum)) * epsilon
            x = x + velocity * step_size

        return x

    def proximal_generator(self, noisy_latent, clean_latent, measurement, sigma, rho):
        return self.mcmc_sample(noisy_latent, clean_latent, measurement, sigma, rho)

    def __call__(self, gt, y_n, record=False, fname=None, save_root=None, inv_transform=None, metrics={}):
        assert inv_transform is not None, "inv_transform cannot be None"
        gt = gt.half()
        y_n = y_n.half()

        log = defaultdict(list)
        cmap = 'gray' if gt.shape[1] == 1 else None

        # get starting latent vector
        # TODO: get this number from vars
        z_latent = torch.randn((1,4,512//8,512//8), device=self.device) * self.rho_initial
        z_latent = z_latent.half()

        # get starting x
        x_latent = z_latent
        x = self.model.decode_image(x_latent)

        # logging
        x_save = inv_transform(x)
        z_save = torch.zeros_like(x_save)
        for name, metric in metrics.items():
            log[name].append(metric(x_save, inv_transform(gt)).item())

        xs_save = torch.cat((inv_transform(gt), x_save), dim=-1).detach().cpu()
        try:
            zs_save = torch.cat((inv_transform(y_n.reshape(*gt.shape)), z_save), dim=-1).detach().cpu()
        except:
            try:
                zs_save = torch.cat((inv_transform(self.operator.A_pinv(y_n).reshape(*gt.shape)), z_save), dim=-1).detach().cpu()
            except:
                zs_save = torch.cat((z_save, z_save), dim=-1).detach().cpu()

        if record:
            log["gt"] = inv_transform(gt).permute(0, 2, 3, 1).squeeze()
            log["x"].append(x_save.permute(0, 2, 3, 1).squeeze().detach())

        samples = []
        iters_count_as_sample = np.linspace(
            self.config.num_burn_in_iters, 
            self.config.num_iters-1, 
            self.config.num_samples_per_run+1, 
            dtype=int
        )[1:]
        assert self.config.num_iters-1 in iters_count_as_sample, "num_iters-1 should be included in iters_count_as_sample"

        rho_values = list(np.logspace(np.log10(self.rho_initial), np.log10(self.rho_final)))

        sub_pbar = tqdm(range(self.config.num_iters))
        for i in sub_pbar:
            rho_iter = rho_values[i]

            # prior step (reverse diffusion)
            x_latent = self.model.sample(z_latent, starting_sigma=rho_iter)
            x = self.model.decode_image(x_latent)

            # likelihood step (langevin dynamics)
            z_latent = self.proximal_generator(z_latent, x_latent, y_n, self.noiser.sigma, rho_iter)
            z0 = self.model.decode_image(z_latent)

            # add noise (forward diffusion)
            if i != len(rho_values)-1:
                z_latent = z_latent + torch.randn_like(z_latent)*rho_values[i+1]
            z = self.model.decode_image(z_latent)
        
            if i in iters_count_as_sample:
                samples.append(x.detach().cpu())

            # logging
            x_save = inv_transform(x)
            z_save = inv_transform(z)
            for name, metric in metrics.items():
                log[name].append(metric(x_save, inv_transform(gt)).item())
            sub_pbar.set_description(f'running PnP-EDM (xrange=[{x.min().item():.2f}, {x.max().item():.2f}], zrange=[{z.min().item():.2f}, {z.max().item():.2f}]) | psnr: {log["psnr"][-1]:.4f}')

            if i % (self.config.num_iters//10) == 0:
                xs_save = torch.cat((xs_save, x_save.detach().cpu()), dim=-1)
                zs_save = torch.cat((zs_save, z_save.detach().cpu()), dim=-1)

            save_grid(torch.cat([x, z0, z]), f"pnpdm_step{i:03}.png")
            
            if record:
                log["x"].append(x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy())

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.plot(log["psnr"])
        plt.title(f'psnr (max): {np.amax(log["psnr"]):.4f}, (last): {log["psnr"][-1]:.4f}')
        plt.subplot(1, 3, 2)
        plt.plot(log["ssim"])
        plt.title(f'ssim (max): {np.amax(log["ssim"]):.4f}, (last): {log["ssim"][-1]:.4f}')
        plt.subplot(1, 3, 3)
        plt.plot(log["lpips"])
        plt.title(f'lpips (min): {np.amin(log["lpips"]):.4f}, (last): {log["lpips"][-1]:.4f}')
        plt.savefig(os.path.join(save_root, 'progress', fname+"_metrics.png"))
        plt.close()

        # logging
        xz_save = torch.cat((xs_save, zs_save), dim=-2).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        plt.imsave(os.path.join(save_root, 'progress', fname+"_x_and_z.png"), xz_save, cmap=cmap)
        np.save(os.path.join(save_root, 'progress', fname+"_log.npy"), log)

        return torch.concat(samples, dim=0).to(self.device)
