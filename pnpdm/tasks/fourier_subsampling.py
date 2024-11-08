import torch
from torch.fft import fft2, ifft2, fftshift
import sigpy.mri
from . import register_operator, LinearOperator

@register_operator(name='fourier_subsampling')
class FourierSubsampling(LinearOperator):
    def __init__(self, channels, img_dim, device) -> None:
        assert channels in [1, 3], 'The number of channels should be either 1 or 3!'
        self.device = device

        # need 3 mask channels or noise is not distributed correctly
        get_mask = lambda: sigpy.mri.poisson((img_dim, img_dim), 30)
        self.mask = torch.tensor([get_mask(), get_mask(), get_mask()], device=device, dtype=torch.float)

        # shift the mask instead of the image
        self.mask = fftshift(self.mask)


    @property
    def display_name(self):
        return 'fourier_subsampling'

    def forward(self, x, **kwargs):
        return fft2(x)*self.mask

    def transpose(self, y, **kwargs):
        return ifft2(y * self.mask.conj())

    def A_pinv(self, y):
        return ifft2(y).real

    def proximal_generator(self, x, y, sigma, rho):
        middle = self.mask / sigma**2 + 1/rho**2

        unit_noise = torch.randn(self.mask.shape, device=self.device)
        noise = ifft2(fft2(unit_noise) / torch.sqrt(middle))

        m = ifft2(fft2(self.transpose(y) / sigma**2 + x/rho**2) / middle)

        return (m + noise).real

    def initialize(self, gt, y):
        return torch.zeros_like(gt)
