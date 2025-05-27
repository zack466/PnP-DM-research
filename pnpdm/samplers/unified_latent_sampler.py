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

def save_latent(latent, target='image.png'):
    latent_saves = latent.transpose(0,1)
    latent_saves = torch.cat([latent_saves, latent_saves, latent_saves], dim=1)
    save_grid(latent_saves, target)


class UnifiedLatent:
    """
    Run our latent version of PnP-DM. This is very similar to the original
    versions with modifications to the likelihood and prior steps which allow
    the algorithm to run in latent space.
    """
    def __init__(self, config, model, operator, noiser, device):
        self.config = config
        self.operator = operator
        self.noiser = noiser
        self.device = device

        self.model = model
        assert hasattr(self.model, "num_steps"), "this sampler requires the model to have a `num_steps` attribute"
        assert hasattr(self.model, "decode_image"), "this sampler requires the model to have a `decode_image` function"
        if not hasattr(self.model, "set_prompt"):
            logging.warning("the model for this sampler has no `set_prompt` function")

        self.likelihood_space = "latent"
        if self.likelihood_space == "latent":
            self.loss = self.latent_loss
        elif self.likelihood_space == "image":
            self.loss = self.image_loss
        else:
            raise ValueError("likelihood_space should be either latent or image")

    def get_rho_schedule(self, num_iters):
        max_allowed = self.model.get_sigma(0)
        min_allowed = self.model.get_sigma(self.model.num_steps - 1)

        if self.config.rho_min is None:
            rho_min = min_allowed
        else:
            rho_min = max(min_allowed, self.config.rho_min)

        if self.config.rho_max is None:
            rho_max = max_allowed
        else:
            rho_max = min(max_allowed, self.config.rho_max)

        if self.config.schedule == "timestep":
            # DAPS schedule matches the model steps
            assert self.model.num_steps == num_iters, "for timestep schedule, num iters should match model steps"
            rho_values = [self.model.get_sigma(i) for i in range(num_iters) if rho_min <= self.model.get_sigma(i) <= rho_max]
            if len(rho_values) < num_iters:
                logging.warning(f"ignoring some noise levels to stay in sigma range {rho_min:.2f}-{rho_max:.2f}")
            return rho_values
        elif self.config.schedule == "exponential":
            # Exponentially decrease rho
            raise ValueError("exponential schedule not yet implemented")
        elif self.config.schedule == "linear":
            rho_schedule = [rho_max - (rho_max - rho_min) * (i / (num_iters - 1)) for i in range(num_iters)]
            return rho_schedule   
        else:
            raise ValueError(f"Unknown rho schedule {self.config.schedule}")

    def get_likelihood_sampler(self):
        if self.config.method == "hmc":
            # HMC sampling code from the DAPS repo
            return self.hmc_sample
        elif self.config.method == "langevin":
            raise NotImplementedError("Langevin sampling not yet implemented")
        elif self.config.method == "optimize":
            # DCDP samples using a pytorch optimizer
            return self.dcdp_sample
        else:
            raise ValueError(f"Unknown likelihood sampler {self.config.method}")

    @property
    def display_name(self):
        return f'{self.config.method}__{self.config.schedule}_schedule__{self.model.__class__.__name__}'

    # latent space
    def latent_loss(self, pred, observation):
        assert pred.shape == (1,4,64,64)
        decoded = self.model.decode_image(pred).float()
        return ((self.operator.forward(decoded) - observation) ** 2).flatten(1).sum(-1)

    # image space
    def image_loss(self, pred, observation):
        assert pred.shape == (1,3,512,512)
        decoded = pred.float()
        return ((self.operator.forward(decoded) - observation) ** 2).flatten(1).sum(-1)

    def get_grad(self, pred, observation):
        pred_tmp = pred.clone().detach().requires_grad_(True)
        loss = self.loss(pred_tmp, observation).sum()
        pred_grad = torch.autograd.grad(loss, pred_tmp)[0]
        pred_grad = pred_grad.to(pred.dtype)
        # clip the gradient
        pred_grad = torch.clamp(pred_grad, -1, 1)
        return pred_grad

    def hmc_sample(self, x0, measurement, sigma, rho):
        lr = self.config.sample_learning_rate
        sample_iters = self.config.sample_num_iters
        momentum = self.config.sample_momentum

        velocity = torch.randn_like(x0)

        x = x0.clone().detach()
        pbar = trange(sample_iters, disable=True)
        for _ in pbar:
            # Langevin step: compute/approximate the score function p(x_0 = x | x_t, y)
            data_fitting_grad = self.get_grad(x, measurement)
            cur_score = -data_fitting_grad / sigma ** 2 + (x0 - x) / rho ** 2

            # update
            epsilon = torch.randn_like(x)
            step_size = np.sqrt(lr)
            velocity = momentum * velocity + step_size * cur_score + np.sqrt(2 * (1 - momentum)) * epsilon
            x = x + velocity * step_size

        return x

    def dcdp_sample(self, x0, measurement, sigma, rho):
        lr = self.config.sample_learning_rate
        sample_iters = self.config.sample_num_iters
        momentum = self.config.sample_momentum

        x = x0.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr=lr, momentum=momentum)

        for i in range(sample_iters):
            loss = self.loss(x, measurement)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return x.detach()
    

    def __call__(self, gt, y_n, record=False, fname=None, save_root=None, inv_transform=None, metrics={}):
        assert inv_transform is not None, "inv_transform cannot be None"
        gt = gt.half()
        y_n = y_n.half()

        log = defaultdict(list)
        cmap = 'gray' if gt.shape[1] == 1 else None

        # get starting latent vector
        z_latent = self.model.get_start()

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

        likelihood_sampler = self.get_likelihood_sampler()
        rho_values = self.get_rho_schedule(self.config.num_iters)

        pbar = tqdm(rho_values)
        for i, rho in enumerate(pbar):
            # prior step (reverse diffusion)
            x_latent = self.model.sample(z_latent, starting_sigma=rho)
            x = self.model.decode_image(x_latent)
            save_latent(x_latent, f"prior_latent{i:03}.png")

            # likelihood step (langevin dynamics)
            if self.likelihood_space == "latent":
                z_latent = likelihood_sampler(x_latent, y_n, self.noiser.sigma, rho)
                z0 = self.model.decode_image(z_latent)
            elif self.likelihood_space == "image":
                z0 = likelihood_sampler(x, y_n, self.noiser.sigma, rho)
                z_latent = self.model.encode_image(z0)
            else:
                raise ValueError("likelihood_space should be latent or image")
            save_latent(z_latent, f"likelihood_before_noise{i:03}.png")

            # add noise (forward diffusion)
            if i != len(rho_values)-1 and not self.config.skip_noising:
                z_latent = z_latent + torch.randn_like(z_latent)*rho_values[i+1]
            z = self.model.decode_image(z_latent)

            if i in iters_count_as_sample:
                samples.append(x.detach().cpu())

            # logging
            x_save = inv_transform(x)
            z_save = inv_transform(z)
            for name, metric in metrics.items():
                log[name].append(metric(x_save, inv_transform(gt)).item())
            pbar.set_description(f'running PnP-EDM (xrange=[{x.min().item():.2f}, {x.max().item():.2f}], zrange=[{z.min().item():.2f}, {z.max().item():.2f}]) | psnr: {log["psnr"][-1]:.4f}')

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
