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
from .denoiser_edm import Denoiser_EDM


class Denoiser_EDM_Latent(Denoiser_EDM):
    def __init__(self, model, device, **kwargs):
        # net is not required since we are overriding __call__()
        super().__init__(model, device, **kwargs)

        # loading in diffusion model
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5")
        self.unet = self.pipeline.unet.to(device)
        self.vae = self.pipeline.vae.to(device)
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder.to(self.device)

    @torch.no_grad()
    def __call__(self, x_noisy, eta):
        # find the smallest t such that sigma(t) < eta
        i_start = torch.min(torch.nonzero(self.sigma(self.t_steps) < eta))

        # Main sampling loop. (starting from t_start with state initialized at x_noisy)
        x_next = x_noisy * self.s(self.t_steps[i_start])

        # switch to latent space
        x_next = self.vae.encode(x_next).latent_dist.sample()

        # text encoding
        prompt = "closeup face of a young asian child"
        text_inputs = self.tokenizer(
            prompt, return_tensors="pt").to(self.device)
        encoder_hidden_states = self.text_encoder(
            **text_inputs).last_hidden_state

        # 0, ..., N-1
        for i, (t_cur, t_next) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])):
            if i < i_start:
                # Skip the steps before i_start.
                continue

            x_cur = x_next
            t_cur = t_cur

            # Euler step.
            lmbd = 2 if self.mode == 'sde' else 1

            unet_timesteps = 0

            denoised = self.unet(
                x_cur / self.s(t_cur), 1, encoder_hidden_states).sample.to(torch.float32)

            d_cur = (lmbd * self.sigma_deriv(t_cur) / self.sigma(t_cur) + self.s_deriv(t_cur) / self.s(t_cur)) * x_cur - \
                lmbd * self.sigma_deriv(t_cur) * self.s(t_cur) / \
                self.sigma(t_cur) * denoised
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Update
            if i != self.num_steps - 1 and self.mode == 'sde':
                n_cur = self.s(t_cur) * torch.sqrt(2 * self.sigma_deriv(t_cur)
                                                   * self.sigma(t_cur)) * torch.randn_like(x_cur)
                x_next += torch.sqrt(t_cur - t_next) * n_cur

        # decode from latent space
        x_next = self.vae.decode(x_next).sample

        return x_next


# class _DenoiserLatentEDM:
#     def __init__(self, device, rho):
#
#         # loading in edm parameters
#         self.rho = 7
#         self.sigma_min = 0.002
#         self.sigma_max = 80
#         self.num_steps = 100
#         self.sigma = lambda t: t
#         self.sigma_deriv = lambda t: 1
#         self.sigma_inv = lambda sigma: sigma
#         self.s = lambda t: 1
#         self.s_deriv = lambda t: 0
#         step_indices = torch.arange(
#             self.num_steps, dtype=torch.float64, device=self.device)
#         sigma_steps = (self.sigma_max ** (1 / rho) + (step_indices / (self.num_steps - 1)) *
#                        (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
#         t_steps = self.sigma_inv(torch.as_tensor(sigma_steps))
#         self.t_steps = torch.cat(
#             [t_steps, torch.zeros_like(t_steps[:1])]).to(self.device)
#
#     def __call__(self, x_noisy, eta):
#         # Find the smallest t such that sigma(t) < eta
#         i_start = torch.min(torch.nonzero(self.sigma(self.t_steps) < eta))
#
#         # Main sampling loop (starting from t_start with state initialized at x_noisy)
#         x_next = x_noisy * self.s(self.t_steps[i_start])
#
#         # switch to latent space
#         x_cur_latent = self.vae.encode(
#             x_noisy / self.s(self.t_steps[i_start])).latent_dist.sample()
#
#         # 0, ..., N-1
#         for i, (t_cur, t_next) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])):
#             if i < i_start:
#                 continue  # Skip the steps before i_start
#
#             # Euler step
#             lmbd = 2
#             sigma_cur = self.sigma(t_cur)
#
#             prompt = "closeup face of a young asian child"
#             text_inputs = self.tokenizer(
#                 prompt, return_tensors="pt").to(self.device)
#             encoder_hidden_states = self.text_encoder(
#                 **text_inputs).last_hidden_state
#
#             with torch.no_grad():
#
#                 c_skip = (0.25) / (sigma_cur ** 2 + 0.25)
#                 c_out = (0.5 * sigma_cur) / (0.25 + sigma_cur ** 2).sqrt()
#                 c_in = 1 / (sigma_cur ** 2 + 0.25).sqrt()
#                 c_noise = 0.25 * np.log(torch.as_tensor(sigma_cur.cpu()))
#                 c_noise = c_noise.to(sigma_cur.device)
#
#                 model_output = self.unet(
#                     (c_in * x_cur_latent).to(torch.float32), c_noise.flatten(), encoder_hidden_states)
#                 model_output = model_output.sample.to(torch.float32)
#                 assert model_output.dtype == torch.float32
#                 denoised = c_skip * x_cur_latent + \
#                     c_out * model_output.to(torch.float32)
#
#                 d_cur = (lmbd * self.sigma_deriv(t_cur) / sigma_cur + self.s_deriv(t_cur) / self.s(t_cur)) * x_cur_latent - \
#                     lmbd * self.sigma_deriv(t_cur) * \
#                     self.s(t_cur) / sigma_cur * denoised
#                 x_next_latent = x_cur_latent + (t_next - t_cur) * d_cur
#
#                 # Update
#                 if i != self.num_steps - 1:
#                     n_cur = self.s(t_cur) * torch.sqrt(2 * self.sigma_deriv(t_cur)
#                                                        * sigma_cur) * torch.randn_like(x_cur_latent)
#                     x_next_latent += torch.sqrt(t_cur - t_next) * n_cur
#
#             del denoised, d_cur, encoder_hidden_states
#             torch.cuda.empty_cache()
#
#             with torch.no_grad():
#                 x_next = self.vae.decode(x_next_latent).sample
#
#         return x_next
