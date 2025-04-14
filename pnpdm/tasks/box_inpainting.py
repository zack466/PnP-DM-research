import torch
from torch.fft import fft2, ifft2, fftshift
import sigpy.mri
from . import register_operator, LinearOperator

@register_operator(name='box_inpainting')
class BoxInpainting(LinearOperator):
    def __init__(self, channels, img_dim, device) -> None:
        assert channels in [1, 3], 'The number of channels should be either 1 or 3!'
        self.device = device

        mask = torch.ones((img_dim, img_dim))
        mask[200:400, 200:400] = 0
        self.mask = torch.stack([mask, mask, mask]).to(device)


    @property
    def display_name(self):
        return 'box_inpainting'

    def forward(self, x, **kwargs):
        return x*self.mask

    def transpose(self, y, **kwargs):
        return y*self.mask

    def A_pinv(self, y):
        return y

    def proximal_generator(self, x, y, sigma, rho):
        middle = self.mask / sigma**2 + 1/rho**2

        unit_noise = torch.randn(self.mask.shape, device=self.device)
        noise = unit_noise / torch.sqrt(middle)

        m = (self.transpose(y) / sigma**2 + x/rho**2) / middle

        return m + noise

    def initialize(self, gt, y):
        return torch.zeros_like(gt)
