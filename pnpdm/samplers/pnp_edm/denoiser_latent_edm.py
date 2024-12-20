from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from torchvision.utils import save_image as tv_save_image
from torchvision.io import read_image as tv_read_image


class StableDiffusionPrecond:
    """
    Precondition the lambdalabs/miniSD-diffusers model to create
    a denoiser as described in the EDM framework. Takes a fixed text
    prompt which describes the expected output image.
    """
    def __init__(self, device, text_prompt, **kwargs):

        self.device = device

        # load diffusion model pipeline and extract components
        self.pipeline = StableDiffusionPipeline.from_pretrained("lambdalabs/miniSD-diffusers")
        self.unet = self.pipeline.unet.to(device)
        self.vae = self.pipeline.vae.to(device)
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder.to(self.device)

        assert self.pipeline.scheduler.beta_schedule == "scaled_linear"

        self.M = self.pipeline.scheduler.config.num_train_timesteps
        self.steps_offset = self.pipeline.scheduler.config.steps_offset
        self.beta_s = self.pipeline.scheduler.config.beta_start * self.M
        self.beta_e = self.pipeline.scheduler.config.beta_end * self.M
        self.beta_d = self.beta_e**0.5 - self.beta_s**0.5

        # setup text prompt embeddings
        prompt = [text_prompt]
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.device))[0]
        self.encoder_hidden_states = torch.cat(
            [uncond_embeddings, text_embeddings])

        self.all_sigma = self.sigma(torch.arange(self.M)).to(self.device)
        self.sigma_min = self.sigma_inv(0).item()
        self.sigma_max = self.sigma_inv(self.M).item()

    def __call__(self, x_noisy, sigma):
        # vp preconditioning with new noise schedule
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1) ** 0.5
        c_noise = self.sigma_inv(sigma)

        with torch.no_grad():
            latent_model_input = torch.cat([x_noisy] * 2)
            unet_out = self.unet(latent_model_input * c_in,
                                 c_noise, self.encoder_hidden_states).sample

        # hopefully classifier free guidance
        noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
        F = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
        return c_skip * x_noisy + c_out * F

    # expects t in [0, 1)
    # Note: this is the alpha's used in the noise derivation, not the noise schedule
    def alpha(self, t):
        return self.beta_s * t + self.beta_s**0.5 * self.beta_d * t**2 + (self.beta_d**2.0 / 3.0) * t**3

    # noise level as a function of t
    # expects t in [0, M)
    def sigma(self, t):
        return (torch.exp(self.alpha((t + self.steps_offset) / self.M)) - 1).sqrt()

    # returns t in [0, M)
    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma, device=self.device).reshape(-1)
        return torch.searchsorted(self.all_sigma, sigma)

    def round_sigma(self, sigma):
        return self.sigma(self.sigma_inv(sigma))


class Denoiser_EDM_Latent():
    def __init__(
        self,
        device,
        text_prompt,
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
        """
        A latent diffusion sampler using the preconditioned model. This can
        also perform image generation starting from partially denoised images.
        """
        self.net = StableDiffusionPrecond(device, text_prompt)
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
        assert mode in [
            'sde', 'pfode'], "Only SDE and PFODE modes are supported."

        # Helper functions for VP & VE noise level schedules.
        def vp_sigma(beta_d, beta_min): return lambda t: (
            np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5

        def vp_sigma_deriv(beta_d, beta_min): return lambda t: 0.5 * \
            (beta_min + beta_d * t) * (self.sigma(t) + 1 / self.sigma(t))
        def vp_sigma_inv(beta_d, beta_min): return lambda sigma: (
            (beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d

        def ve_sigma(t): return t.sqrt()
        def ve_sigma_deriv(t): return 0.5 / t.sqrt()
        def ve_sigma_inv(sigma): return sigma ** 2

        # Select default noise level range based on the specified time step discretization.
        if sigma_min is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
            sigma_min = {'vp': vp_def, 've': 0.02,
                         'iddpm': 0.002, 'edm': 0.002}[discretization]
        if sigma_max is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
            sigma_max = {'vp': vp_def, 've': 100,
                         'iddpm': 81, 'edm': 80}[discretization]

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Compute corresponding betas for VP.
        vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s -
                         np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

        # Define time steps in terms of noise level.
        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=device)
        if discretization == 'vp':
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif discretization == 've':
            orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 /
                                                sigma_max ** 2) ** (step_indices / (num_steps - 1)))
            sigma_steps = ve_sigma(orig_t_steps)
        elif discretization == 'iddpm':
            u = torch.zeros(M + 1, dtype=torch.float64, device=device)
            def alpha_bar(j): return (
                0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=device):  # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) /
                            alpha_bar(j)).clip(min=C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[(
                (len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
        else:
            assert discretization == 'edm'
            sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
                           * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

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
            self.s_deriv = lambda t: - \
                self.sigma(t) * self.sigma_deriv(t) * (self.s(t) ** 3)
        else:
            assert scaling == 'none'
            self.s = lambda t: 1
            self.s_deriv = lambda t: 0

        # Compute final time steps based on the corresponding noise levels.
        t_steps = self.sigma_inv(self.net.round_sigma(sigma_steps))
        self.t_steps = torch.cat(
            [t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # encode image to latent space, never use torch grad
    def encode_image(self, img):
        with torch.no_grad():
            encoded = self.net.vae.encode(img).latent_dist.sample() * 0.18215
            return encoded

    # decode latent to image space, we need the grad for our likelihood step
    def decode_image(self, latents):
        decoded = self.net.vae.decode(latents / 0.18215).sample
        return decoded

    @torch.no_grad()
    def __call__(self, z_noisy, eta):
        # find the smallest t such that sigma(t) < eta
        i_start = torch.min(torch.nonzero(self.sigma(self.t_steps) < eta))

        # Main sampling loop. (starting from t_start with state initialized at x_noisy)
        x_next = z_noisy * self.s(self.t_steps[i_start])

        # uncomment this and set eta to inf to automatically run from pure noise every time
        # x_next = torch.randn(1, 4, 256//8, 256//8, device=self.device) * \
            # self.s(self.t_steps[i_start]) * self.sigma(self.t_steps[i_start])

        # 0, ..., N-1
        for i, (t_cur, t_next) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])):
            if i < i_start:
                # Skip the steps before i_start.
                continue

            x_cur = x_next
            t_cur = t_cur

            # Euler step.
            lmbd = 2 if self.mode == 'sde' else 1
            denoised = self.net(x_cur / self.s(t_cur),
                                self.sigma(t_cur)).to(torch.float32)
            d_cur = (lmbd * self.sigma_deriv(t_cur) / self.sigma(t_cur) + self.s_deriv(t_cur) / self.s(t_cur)) * x_cur - \
                lmbd * self.sigma_deriv(t_cur) * self.s(t_cur) / \
                self.sigma(t_cur) * denoised
            x_next = x_cur + (t_next - t_cur) * d_cur

            # self.save_image(self.decode_image(x_next), f"test-{i:04}.png")

            # Update
            if i != self.num_steps - 1 and self.mode == 'sde':
                n_cur = self.s(t_cur) * torch.sqrt(2 * self.sigma_deriv(t_cur)
                                                   * self.sigma(t_cur)) * torch.randn_like(x_cur)
                x_next += torch.sqrt(t_cur - t_next) * n_cur

        return x_next

    # save image which is already scaled from -1 to 1
    def save_image(self, img, path):
        tv_save_image((img / 2 + 0.5).clamp(0, 1), path)

    # read image and scale from -1 to 1
    def read_image(self, path):
        img = torch.tensor(tv_read_image(path) /
                         255.0, device=self.device)[None, :3, :, :]
        return img*2-1

if __name__ == "__main__":
    device = torch.device("cuda")
    model = Denoiser_EDM_Latent(device, "boy with a tree behind him", num_steps=100)

    x_clean = model.read_image("images/00014.png")
    z_clean = model.encode_image(x_clean)

    sigma = 1
    z_noisy = z_clean + sigma * torch.randn_like(z_clean)
    model.save_image(model.decode_image(z_noisy), "noised.png")
    z_denoised = model(z_noisy, sigma)
    x_denoised = model.decode_image(z_denoised)

    model.save_image(x_denoised, "denoised.png")
