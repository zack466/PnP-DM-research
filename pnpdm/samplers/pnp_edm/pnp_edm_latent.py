import torch, os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

from .denoiser_latent_edm import Denoiser_EDM_Latent

class PnPEDMLatent:
    def __init__(self, config, model, operator, noiser, device):
        self.config = config
        self.model = model
        self.operator = operator
        self.noiser = noiser
        self.device = device
        if config.mode == 'vp':
            self.edm = Denoiser_EDM_Latent(model, device, **config.common_kwargs, **config.vp_kwargs, mode='pfode')
        elif config.mode == 've':
            self.edm = Denoiser_EDM_Latent(model, device, **config.common_kwargs, **config.ve_kwargs, mode='pfode')
        elif config.mode == 'iddpm':
            self.edm = Denoiser_EDM_Latent(model, device, **config.common_kwargs, **config.iddpm_kwargs, mode='pfode')
        elif config.mode == 'edm':
            self.edm = Denoiser_EDM_Latent(model, device, **config.common_kwargs, **config.edm_kwargs, mode='pfode')
        elif config.mode == 'vp_sde':
            self.edm = Denoiser_EDM_Latent(model, device, **config.common_kwargs, **config.vp_kwargs, mode='sde')
        elif config.mode == 've_sde':
            self.edm = Denoiser_EDM_Latent(model, device, **config.common_kwargs, **config.ve_kwargs, mode='sde')
        elif config.mode == 'iddpm_sde':
            self.edm = Denoiser_EDM_Latent(model, device, **config.common_kwargs, **config.iddpm_kwargs, mode='sde')
        elif config.mode == 'edm_sde':
            self.edm = Denoiser_EDM_Latent(model, device, **config.common_kwargs, **config.edm_kwargs, mode='sde')
        else:
            raise NotImplementedError(f"Mode {self.config.mode} is not implemented (must be latent_sde for pnp_edm_latent)")

    @property
    def display_name(self):
        return f'pnp-edm-latent-{self.config.mode}-rho0={self.config.rho}-rhomin={self.config.rho_min}'

    def proximal_generator(self, x, y, sigma, rho, gamma=2e-4, num_iters=200):
        z = x
        z.requires_grad = True
        for _ in range(num_iters):
            # forward operator is A(D(z))
            data_fit = (self.operator.forward(self.edm.decode_image(z)) - y).norm()**2 / (2*sigma**2)
            grad = torch.autograd.grad(outputs=data_fit, inputs=z)[0]
            z = z - gamma * grad - (gamma/rho**2) * (z - x) #+ np.sqrt(2*gamma) * torch.randn_like(x)
        return z.type(torch.float32) + rho * torch.randn_like(x)

    def __call__(self, gt, y_n, record=False, fname=None, save_root=None, inv_transform=None, metrics={}):
        assert inv_transform is not None, "inv_transform cannot be None"

        log = defaultdict(list)
        cmap = 'gray' if gt.shape[1] == 1 else None
        x = self.operator.initialize(gt, y_n)
        x_latent = self.edm.encode_image(x)

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
        sub_pbar = tqdm(range(self.config.num_iters))
        for i in sub_pbar:
            rho_iter = self.config.rho * (self.config.rho_decay_rate**i)
            rho_iter = max(rho_iter, self.config.rho_min)

            # likelihood step
            z_latent = self.proximal_generator(x_latent, y_n, self.noiser.sigma, rho_iter)
            z = self.edm.decode_image(z_latent)
        
            # prior step
            x_latent = self.edm(z_latent, rho_iter)
            x = self.edm.decode_image(x_latent)

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

            self.edm.save_image(x, "current_x.png")
            self.edm.save_image(z, "current_z.png")
            # plt.imsave(os.path.join(save_root, 'progress', fname+f"x-{i}.png"), x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy(), cmap=cmap)
            # plt.imsave(os.path.join(save_root, 'progress', fname+f"z-{i}.png"), z_save.permute(0, 2, 3, 1).squeeze().cpu().numpy(), cmap=cmap)
            
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

# class Mode:
#     mode = "edm_sde"
#     common_kwargs = {}
#     edm_kwargs = {}
#
# if __name__ == "__main__":
#     device = torch.device("cuda")
#     operator = GaussialBlurCircular(61, 7.0, 3, 256, device)
#     noiser = GaussianNoise(0.05)
#     pnp = PnPEDMLatent(Mode(), None, operator, noiser, device)
#
#     gt = pnp.edm.read_image("images/00003.png")
#     y = noiser.forward(operator.forward(gt))
#
#     encoded = pnp.edm.read_image("images/00003.png")
#     encoded = pnp.edm.encode_image(encoded)
#     encoded = torch.zeros_like(encoded)
#
#     rho = 2
#     likelihood = pnp.proximal_generator(encoded, y, noiser.sigma, rho)
#
#     pnp.edm.save_image(pnp.edm.encode_image(gt) + torch.randn_like(encoded)*rho, "encoded.png")
#     pnp.edm.save_image(likelihood, "likelihood.png")
