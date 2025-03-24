import torch, os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from collections import defaultdict
from torchvision.utils import save_image
from PIL import Image

from .denoiser_latent_edm import StableDiffusionModel

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

class PnPEDMLatent:
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

        # self.edm = Denoiser_EDM_Latent(*args, **kwargs)
        self.edm = StableDiffusionModel(device=self.device, prompt=config.text_prompt)

    @property
    def display_name(self):
        return f'pnp-edm-latent-{self.config.mode}-rho0={self.config.rho}-rhomin={self.config.rho_min}'

    def loss(self, pred, observation):
        decoded = self.edm.decode_image(pred).float()
        return ((self.operator.forward(decoded) - observation) ** 2).flatten(1).sum(-1)

    def get_grad(self, pred, observation, return_loss=False):
        pred_tmp = pred.clone().detach().requires_grad_(True)
        loss = self.loss(pred_tmp, observation).sum()
        pred_grad = torch.autograd.grad(loss, pred_tmp)[0]
        pred_grad = pred_grad.to(pred.dtype)
        # clip the gradient
        pred_grad = torch.clamp(pred_grad, -1, 1)
        if return_loss:
            return pred_grad, loss
        else:
            return pred_grad


    def mcmc_sample(self, xt, x0hat, measurement, sigma, rho):
        lr = 1e-4
        num_steps = 30
        momentum = 0.45

        velocity = torch.randn_like(x0hat)
        prior_score = (x0hat - xt).detach() / rho ** 2

        x = x0hat.clone().detach()
        pbar = trange(num_steps)
        for _ in pbar:
            # Langevin step: compute/approximate the score function p(x_0 = x | x_t, y)
            data_fitting_grad, data_fitting_loss = self.get_grad(x, measurement, return_loss=True)
            data_term = -data_fitting_grad / sigma ** 2
            xt_term = (xt - x) / rho ** 2
            cur_score, fitting_loss = data_term + xt_term + prior_score, data_fitting_loss
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
        z_latent = self.edm.get_start(1)

        # get starting x
        x_latent = z_latent
        x = self.edm.decode_image(x_latent)

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

        rho_values = [self.edm.get_sigma(i) for i in range(self.edm.num_steps)]
        self.config.num_iters = len(rho_values)

        sub_pbar = tqdm(range(self.config.num_iters))
        for i in sub_pbar:
            # rho_iter = self.config.rho * (self.config.rho_decay_rate**i)
            # rho_iter = max(rho_iter, self.config.rho_min)
            rho_iter = rho_values[i]

            # prior step (reverse diffusion)
            x_latent = self.edm.sample(z_latent, starting_sigma=rho_iter)
            x = self.edm.decode_image(x_latent)

            # likelihood step (langevin dynamics)
            z_latent = self.proximal_generator(z_latent, x_latent, y_n, self.noiser.sigma, rho_iter)
            z0 = self.edm.decode_image(z_latent)

            # add noise (forward diffusion)
            if i != len(rho_values)-1:
                z_latent = z_latent + torch.randn_like(z_latent)*rho_values[i+1]
            z = self.edm.decode_image(z_latent)
        
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

            # save_grid(torch.cat([x, z0, z]), f"pnpdm_step{i:03}.png")
            
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
