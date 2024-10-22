import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import register_operator, LinearOperator

@register_operator(name='inpainting')
class Inpainting(LinearOperator):
    def __init__(self, kernel_size, intensity, channels, img_dim, device) -> None:
        assert channels in [1, 3], 'The number of channels should be either 1 or 3!'

    @property
    def display_name(self):
        return 'inpainting'

    def forward(self, x, **kwargs):
        pass

    def transpose(self, y):
        pass
    
