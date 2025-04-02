from .edm.edm import create_edm_from_unet_adm
from .sd_wrappers.daps_sd_wrapper import DapsSDWrapper

def get_model(name: str, device: str, **kwargs):
    if name == 'edm_from_unet_adm':
        model = create_edm_from_unet_adm(**kwargs)
        model = model.to(device)
        model.eval()
        return model
    elif name == 'edm_prior':
        # TODO: add this back
        return None
    elif name == 'daps_sd_wrapper':
        return DapsSDWrapper(device=device, **kwargs)
    else:
        raise NameError(f"Model {name} is not defined.")
