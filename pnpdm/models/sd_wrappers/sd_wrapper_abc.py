from abc import ABC, abstractmethod

class SDWrapper(ABC):
    @abstractmethod
    def __init__(self, resolution, num_steps, device):
        raise NotImplementedError

    @abstractmethod
    def set_prompt(self, prompt: str):
        raise NotImplementedError

    @abstractmethod
    def encode_image(self, x0):
        raise NotImplementedError

    @abstractmethod
    def decode_image(self, z0):
        raise NotImplementedError

    @abstractmethod
    def sample(self, z_start, starting_sigma):
        raise NotImplementedError

    @abstractmethod
    def get_start(self):
        raise NotImplementedError

    def get_sigma(self, timestep):
        raise NotImplementedError
