from .pnp_edm.pnp_edm import PnPEDM, PnPEDMBatch
from .daps_hmc import DapsHMC
from .pnp_edm.pnp_edm_bh import PnPEDMBH, PnPEDMBHBatch
from .pnp_edm.pnp_edm_int import PnPEDMINT, PnPEDMBatchINT
from .pnpdm_latent import PnPDMLatent

def get_sampler(config, model, operator, noiser, device):
    if config.name == 'pnp_edm':
        return PnPEDM(config, model, operator, noiser, device)
    elif config.name == 'pnp_edm_int':
        return PnPEDMINT(config, model, operator, noiser, device)
    elif config.name == 'pnp_edm_batch':
        return PnPEDMBatch(config, model, operator, noiser, device)
    elif config.name == 'pnp_edm_bh':
        return PnPEDMBH(config, model, operator, noiser, device)
    elif config.name == 'pnp_edm_batch_bh':
        return PnPEDMBHBatch(config, model, operator, noiser, device)
    elif config.name == 'pnpdm_latent':
        return PnPDMLatent(config, model, operator, noiser, device)
    elif config.name == 'daps_hmc':
        return DapsHMC(config, model, operator, noiser, device)
    else:
        raise NameError(f"Sampler {config.name} is not defined.")
