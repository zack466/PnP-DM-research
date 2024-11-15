# https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main
# - See files and versions, see json configs about architectures about UNet/VAE/Encoder
# - All these thigns we need to load to our repo
# - There's no script or code that does it, but the API
# - `diffuser` package has DDPM pipeline
# - pipeline_utils
#     - from_pretrained method
#     - Line like 780 or something seems important
#     - Loading in 868
#     - Take necessary components in this functions to PNPDM
#         - VAE, UNet
# - Easier way is probably to install diffuser package and write another pipeline that inherits Diffusion Pipeline
#     - Then modify some things
#     - Make "EDMDenoiserPipeline" from DDPMPipeline
#     - Call function from this should be your prior step
#         - image input and rho :(
#         - Right now, makes random vector and goes through entire diffusion process
#         - When you do call, make it the prior step of PnPDM
#         - Add arguments z and rho, migrate call step from PnPDM to this call function
# - Run with PnPDM to just test it working then integrate preconditioning to make sure it works

# preconditionng on unet
# [for loop} replace with algorhtm 3
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from torchvision.utils import save_image
from torchvision.io import read_image


class StableDiffusionPrecond:
    def __init__(self, device, **kwargs):

        self.device = device

        # loading in diffusion model
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5")
        self.unet = self.pipeline.unet.to(device)
        self.vae = self.pipeline.vae.to(device)
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder.to(self.device)

        self.alphas = 1.0 - self.pipeline.scheduler.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)

        # text encoding?
        prompt = ["face of an asian child about 2 years old, as a high quality photograph"]
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        self.encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

        self.sigma_min = self.sigma_inv(0).item()
        self.sigma_max = self.sigma_inv(len(self.alphas)).item()

    def __call__(self, x_noisy, sigma):
        # ???
        print(sigma)
        t = self.sigma_inv(sigma)

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1) ** 0.5
        c_noise = t
        print("t =", c_noise)

        with torch.no_grad():
            latent_model_input = torch.cat([x_noisy] * 2)
            unet_out = self.unet(latent_model_input * c_in, c_noise, self.encoder_hidden_states).sample

        # hopefully classifier free guidance
        noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
        F = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
        return c_skip * x_noisy + c_out * F

    def sigma(self, t):
        return torch.sqrt(1 - self.alphas_cumprod[t]) / torch.sqrt(self.alphas_cumprod[t])

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma, device=self.device).reshape(-1)
        all_sigma = self.sigma(torch.arange(len(self.alphas)))
        return torch.argmin(torch.abs(all_sigma[None, :] - sigma[:, None]), dim=1)

    def round_sigma(self, sigma):
        return self.sigma(self.sigma_inv(sigma))


class Denoiser_EDM_Latent():
    def __init__(
        self,
        throwaway,
        device,
        num_steps=18,
        sigma_min=None,
        sigma_max=None,
        rho=7,
        solver='euler',
        discretization='edm',
        schedule='linear',
        scaling='none',
        epsilon_s=1e-3,
        C_1=0.001,
        C_2=0.008,
        M=1000,
        alpha=1,
        S_churn=0,
        S_min=0,
        S_max=float('inf'),
        S_noise=1,
        mode='sde'
    ):
        self.net = StableDiffusionPrecond(device)
        self.device = device
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.solver = solver
        self.discretization = discretization
        self.schedule = schedule
        self.scaling = scaling
        self.epsilon_s = epsilon_s
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.alpha = alpha
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.mode = mode

        assert solver in ['euler'], "Only Euler solver is supported."
        assert discretization in ['vp', 've', 'iddpm', 'edm']
        assert schedule in ['vp', 've', 'linear']
        assert scaling in ['vp', 'none']
        assert mode in ['sde', 'pfode'], "Only SDE and PFODE modes are supported."

        # Helper functions for VP & VE noise level schedules.
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (self.sigma(t) + 1 / self.sigma(t))
        vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
        ve_sigma = lambda t: t.sqrt()
        ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
        ve_sigma_inv = lambda sigma: sigma ** 2

        # Select default noise level range based on the specified time step discretization.
        if sigma_min is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
            sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
        if sigma_max is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
            sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Compute corresponding betas for VP.
        vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

        # Define time steps in terms of noise level.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        if discretization == 'vp':
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif discretization == 've':
            orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
            sigma_steps = ve_sigma(orig_t_steps)
        elif discretization == 'iddpm':
            u = torch.zeros(M + 1, dtype=torch.float64, device=device)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=device): # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
        else:
            assert discretization == 'edm'
            sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        # Define noise level schedule.
        if schedule == 'vp':
            self.sigma = vp_sigma(vp_beta_d, vp_beta_min)
            self.sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
            self.sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
        elif schedule == 've':
            self.sigma = ve_sigma
            self.sigma_deriv = ve_sigma_deriv
            self.sigma_inv = ve_sigma_inv
        else:
            assert schedule == 'linear'
            self.sigma = lambda t: t
            self.sigma_deriv = lambda t: 1
            self.sigma_inv = lambda sigma: sigma

        # Define scaling schedule.
        if scaling == 'vp':
            self.s = lambda t: 1 / (1 + self.sigma(t) ** 2).sqrt()
            self.s_deriv = lambda t: -self.sigma(t) * self.sigma_deriv(t) * (self.s(t) ** 3)
        else:
            assert scaling == 'none'
            self.s = lambda t: 1
            self.s_deriv = lambda t: 0
        
        # Compute final time steps based on the corresponding noise levels.
        t_steps = self.sigma_inv(self.net.round_sigma(sigma_steps))
        self.t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    def encode_image(self, img):
        with torch.no_grad():
            img = img * 2 - 1
            encoded = self.net.vae.encode(img).latent_dist.sample() * 0.18215
            return encoded

    def decode_image(self, latents):
        with torch.no_grad():
            decoded = self.net.vae.decode(latents / 0.18215).sample
            decoded = (decoded / 2 + 0.5).clamp(0,1)
            return decoded

    @torch.no_grad()
    def __call__(self, x_noisy, eta):
        i_start = torch.min(torch.nonzero(self.sigma(self.t_steps) < eta)) # find the smallest t such that sigma(t) < eta

        # Main sampling loop. (starting from t_start with state initialized at x_noisy)
        x_next = x_noisy * self.s(self.t_steps[i_start])
        x_next = self.encode_image(x_next)

        x_next = torch.randn(1, 4, 512//8, 512//8, device=self.device) * self.s(self.t_steps[i_start]) * self.sigma(self.t_steps[i_start])
        
        for i, (t_cur, t_next) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])): # 0, ..., N-1
            if i < i_start:
                # Skip the steps before i_start.
                continue

            x_cur = x_next
            t_cur = t_cur

            # Euler step.
            lmbd = 2 if self.mode == 'sde' else 1
            denoised = self.net(x_cur / self.s(t_cur), self.sigma(t_cur)).to(torch.float32)
            d_cur = (lmbd * self.sigma_deriv(t_cur) / self.sigma(t_cur) + self.s_deriv(t_cur) / self.s(t_cur)) * x_cur - \
                     lmbd * self.sigma_deriv(t_cur) * self.s(t_cur) / self.sigma(t_cur) * denoised 
            x_next = x_cur + (t_next - t_cur) * d_cur

            save_image(self.decode_image(x_next), f"test-{i:04}.png")

            # Update
            if i != self.num_steps - 1 and self.mode == 'sde':
                n_cur = self.s(t_cur) * torch.sqrt(2 * self.sigma_deriv(t_cur) * self.sigma(t_cur)) * torch.randn_like(x_cur)
                x_next += torch.sqrt(t_cur - t_next) * n_cur


        x_next = self.decode_image(x_next)
        return x_next


if __name__ == "__main__":
    device = torch.device("cuda")
    model = Denoiser_EDM_Latent(None, device, num_steps=100)
    xlist = model(torch.randn(1, 3, 512, 512, device=device), float("inf"))
    exit()

    D = StableDiffusionPrecond(device)
    encode = lambda x: D.vae.encode(2*x - 1).latent_dist.sample() * 0.18215
    decode = lambda z: D.vae.decode(z / 0.18215).sample / 2 + 0.5

    x = torch.tensor(read_image("images/00003.png") / 255.0, device=device)[None, :3, :, :]

    z = encode(x)
    noise = torch.randn(1, 4, 256//8, 256//8, device=device) * 1
    z_noisy = z + noise

    z_clean = D(z_noisy, 1)

    save_image(decode(z_noisy), f"noisy.png")
    save_image(decode(z_clean), f"clean.png")
