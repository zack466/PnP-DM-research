# TODO: add pnpdm sampler

python posterior_sample.py \
    +data=images_with_prompts \
    +task=super_resolution_svd \
    +model=pnpdm_sd_wrapper \
    +sampler=pnpdm \
    add_exp_name=pnpdm_superres

python posterior_sample.py \
    +data=images_with_prompts \
    +task=gaussian_deblur_circ \
    +model=pnpdm_sd_wrapper \
    +sampler=pnpdm \
    add_exp_name=pnpdm_deblur

python posterior_sample.py \
    +data=images_with_prompts \
    +task=box_inpainting \
    +model=pnpdm_sd_wrapper \
    +sampler=pnpdm \
    add_exp_name=pnpdm_box
