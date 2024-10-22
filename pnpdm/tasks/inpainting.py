import torch
from . import register_operator, LinearOperator

@register_operator(name='inpainting')
class Inpainting(LinearOperator):
    def __init__(self, channels, img_dim, device) -> None:
        assert channels in [1, 3], 'The number of channels should be either 1 or 3!'
        self.device = device

        # box mask
        # self.mask = torch.ones(1, channels, img_dim, img_dim, device=device)
        # self.mask[:, :, 3*img_dim//8 : 5*img_dim//8, 3*img_dim//8 : 5*img_dim//8] = 0

        # random mask
        p = 0.9
        self.mask = torch.rand(1, channels, img_dim, img_dim, device=device)
        self.mask[self.mask <= p] = 0
        self.mask[self.mask > p] = 1


    @property
    def display_name(self):
        return 'inpainting'

    # compute Ax
    def forward(self, x, **kwargs):
        return x * self.mask

    # compute A^Ty
    def transpose(self, y, **kwargs):
        return y * self.mask
    
    # likelihood of measurement
    def proximal_generator(self, x, y, sigma, rho):
        inv_lambda = 1 / (self.mask / sigma**2 + 1/rho**2)
        m = inv_lambda * (self.transpose(y) / sigma**2 + x/rho**2)
        noise = torch.randn(x.shape, device=self.device) * torch.sqrt(inv_lambda)
        return m + noise

    def initialize(self, gt, y):
        return torch.zeros_like(gt)
