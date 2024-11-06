from diffusers.pipelines.pipeline_utils import DiffusionPipeline

class DenoiserLatentEDM(DiffusionPipeline):

    def __init__(self):
        pass
        # Get config from some file
        # Run latent step

    def __call__(self, **kwargs):
        pass
        # Copy DiffusionPipelines' __call__ but with preconditioning stuff :(

# TODO

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
