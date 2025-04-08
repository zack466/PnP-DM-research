from diffusers import StableDiffusionPipeline
import torch
from .edm_schedules import SDSchedule, VPSchedule, EDMSchedule
from .sd_wrapper_abc import SDWrapper

class StableDiffusionPrecond:
    def __init__(self, pipeline: StableDiffusionPipeline, size, num_steps, device):
        self.device = device
        self.pipeline = pipeline
        self.pipeline.scheduler.set_timesteps(num_steps)
        self.schedule = SDSchedule(pipeline.scheduler.config, device=device)
        self.encoder_hidden_states = None

        self.size = size
        self.num_steps = num_steps

    def encode_text(self, text_prompt):
        prompt = [text_prompt]
        text_input = self.pipeline.tokenizer(
            prompt, padding="max_length", max_length=self.pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = self.pipeline.text_encoder(
            text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.pipeline.tokenizer(
            [""], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.pipeline.text_encoder(
            uncond_input.input_ids.to(self.device))[0]
        encoder_hidden_states = torch.cat(
            [uncond_embeddings, text_embeddings])
        return encoder_hidden_states

    def set_prompt(self, text_prompt):
        self.encoder_hidden_states = self.encode_text(text_prompt)
        
    def clear_prompt(self):
        self.encoder_hidden_states = None
        
    def encode_image(self, img):
        """
        Expects an image of shape (1, 3, N, N), scaled from -1 to 1.
        """
        assert img.shape == torch.Size([1, 3, self.size, self.size]), \
            f"image size is {img.shape} but expected to be 1x3x{self.size}x{self.size}"
        with torch.no_grad():
            encoded = self.pipeline.vae.encode(img).latent_dist.sample() * 0.18215
            return encoded
    
    # decode latent to image space, we need the grad for our likelihood step
    def decode_image(self, latents):
        """
        Expects an image of shape (1, 3, N, N), scaled from -1 to 1.
        """
        decoded = self.pipeline.vae.decode(latents.half() / 0.18215).sample
        return decoded
        
    def denoise_step(self, x_noisy, sigma, encoded_text=None):
        """
        Performs a single step of denoising on a preconditioned stable diffusion pipeline.
        """
        # vp preconditioning with new noise schedule
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1) ** 0.5
        c_noise = self.schedule.sigma_inv(sigma.to(self.device))
    
        # we expect precond.sigma(c_noise) > sigma by a small amount due to quantization error
        missing_noise = torch.sqrt(self.schedule.sigma(c_noise) ** 2 - sigma ** 2).to(x_noisy.dtype)
        x_noisy = x_noisy + missing_noise * torch.randn_like(x_noisy, dtype=x_noisy.dtype)

        encoded_text = self.encoder_hidden_states if encoded_text is None else encoded_text
    
        with torch.no_grad():
            latent_model_input = torch.cat([x_noisy] * 2)
            unet_out = self.pipeline.unet(latent_model_input * c_in,
                                 c_noise, encoder_hidden_states=encoded_text).sample
    
        # hopefully classifier free guidance
        noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
        F = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
        return c_skip * x_noisy + c_out * F


class EDM_SD_Wrapper(SDWrapper):
    def __init__(self, num_steps, device, resolution=512, model_id="sd-legacy/stable-diffusion-v1-5", mode="sde"):
        self.device = device
        self.res = resolution
        self.num_steps = num_steps
        self.device = device

        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        self.pipeline.unet = self.pipeline.unet.half()
        self.pipeline.vae = self.pipeline.vae.half()
        self.pipeline.text_encoder = self.pipeline.text_encoder.half()
        self.pipeline = self.pipeline.to(device)
        self.precond = StableDiffusionPrecond(self.pipeline, resolution, num_steps, device)
        self.mode = mode

        self.schedule = VPSchedule(0.02, 10, num_steps=num_steps, device=device)

    @torch.no_grad()
    def sample(self, z_start, starting_sigma):
        """
        Uses the PnP-DM sampling algorithm to denoise a noisy latent with a
        preconditioned stable diffusion model.
        """
        t_steps = self.schedule.sigma_inv(self.schedule.sigma_steps)
        
        # find the smallest t such that sigma(t) < eta
        i_start = torch.min(torch.nonzero(self.schedule.sigma(t_steps) <= starting_sigma))

        x_next = z_start * self.schedule.s(t_steps[i_start])

        # 0, ..., N-1
        for i, (t, tn) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            if i < i_start:
                # Skip the steps before i_start.
                continue

            x_cur = x_next

            # Euler step.
            lmbd = 2 if self.mode == 'sde' else 1

            denoised = self.precond.denoise_step(x_cur / self.schedule.s(t), self.schedule.sigma(t))
            
            d_cur = (lmbd * self.schedule.sigma_deriv(t) / self.schedule.sigma(t) + self.schedule.s_deriv(t) / self.schedule.s(t)) * x_cur - \
                lmbd * self.schedule.sigma_deriv(t) * self.schedule.s(t) / \
                self.schedule.sigma(t) * denoised
            x_next = x_cur + (tn - t) * d_cur
            
            # Update
            if i != self.schedule.num_steps - 1 and self.mode == 'sde':
                n_cur = self.schedule.s(t) * torch.sqrt(2 * self.schedule.sigma_deriv(t)
                                                   * self.schedule.sigma(t)) * torch.randn_like(x_cur)
                x_next += torch.sqrt(t - tn) * n_cur

        return x_next

    def set_prompt(self, prompt: str):
        self.precond.set_prompt(prompt)

    def encode_image(self, x0):
        return self.precond.encode_image(x0)

    def decode_image(self, z0):
        return self.precond.decode_image(z0)

    def get_start(self):
        sigma_max = self.schedule.sigma_steps.max().to(self.device)
        val = torch.randn((1, 4, self.res//8, self.res//8), device=self.device) * sigma_max
        return val.half()

    def get_sigma(self, timestep):
        return self.schedule.sigma_steps[timestep]
