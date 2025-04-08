import torch
import numpy as np

# Convert Stable Diffusion scaled linear schedule (PNDM) to EDM formulation (in terms of sigma)
# https://arxiv.org/pdf/2206.00364
# Appendix C.1
class SDSchedule:
    def __init__(self, config, device):
        self.device = device
        self.steps_offset = config.steps_offset
        self.train_timesteps = config.num_train_timesteps
        
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py#L133
        self.beta_start, self.beta_end = config.beta_start * self.train_timesteps, config.beta_end * self.train_timesteps
        self.beta_d = self.beta_end**0.5 - self.beta_start**0.5
        self.all_sigma = self.sigma(torch.arange(self.train_timesteps)).to(device)
        
    # expects t in [0, 1]
    def alpha(self, t):
        return self.beta_start * t + self.beta_start**0.5 * self.beta_d * t**2 + (self.beta_d**2.0 / 3.0) * t**3
    
    # expects t in [0, 1000]
    def sigma(self, t):
        return (torch.exp(self.alpha((t + self.steps_offset) / self.train_timesteps)) - 1) ** 0.5
    
    # returns t in [0, M)
    def sigma_inv(self, s):
        # subtract a small amount to prevent accidental rounding up
        return torch.searchsorted(self.all_sigma - 1e-6, s.reshape(-1))
    
    def round_sigma(self, s):
        return self.sigma(self.sigma_inv(s))

# Now that the network is pre-conditioned, we can give it any noise schedule
# (e.g. from VP, VE, EDM, etc) in terms of sigma and denoise using Stable
# Diffusion according to that schedule (no matter the original formulation).
class VPSchedule:
    def __init__(self, beta_min, beta_d, num_steps, device, epsilon_s=1e-3):
        self.device = device
        self.num_steps = num_steps
            
        def vp_sigma(beta_d, beta_min): return lambda t: (
            np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
            
        sigma_min = vp_sigma(beta_d, beta_min)(t=epsilon_s)
        sigma_max = vp_sigma(beta_d, beta_min)(t=1)

        def vp_sigma_deriv(beta_d, beta_min): return lambda t: 0.5 * \
            (beta_min + beta_d * t) * (self.sigma(t) + 1 / self.sigma(t))
            
        def vp_sigma_inv(beta_d, beta_min): return lambda sigma: (
            (beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
        
        vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s -
                         np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        self.sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)

        self.s = lambda t: 1 / (1 + self.sigma(t) ** 2).sqrt()
        self.s_deriv = lambda t: - \
            self.sigma(t) * self.sigma_deriv(t) * (self.s(t) ** 3)
        
        self.sigma = vp_sigma(vp_beta_d, vp_beta_min)
        self.sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        self.sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)

class EDMSchedule:
    def __init__(self, sigma_min, sigma_max, rho, num_steps, device):
        self.device = device
        self.num_steps = num_steps
        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=device)
        self.sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
                       * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        
        self.sigma = lambda t: t
        self.sigma_deriv = lambda t: 1
        self.sigma_inv = lambda sigma: sigma
    
        self.s = lambda t: 1
        self.s_deriv = lambda t: 0
