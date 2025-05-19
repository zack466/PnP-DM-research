python posterior_sample.py \
    +data=images_with_prompts \
    +task=super_resolution_svd \
    +model=daps_sd_wrapper \
    +sampler=dcdp \
    add_exp_name=dcdp_superres

python posterior_sample.py \
    +data=images_with_prompts \
    +task=gaussian_deblur_circ \
    +model=daps_sd_wrapper \
    +sampler=dcdp \
    add_exp_name=dcdp_deblur

python posterior_sample.py \
    +data=images_with_prompts \
    +task=box_inpainting \
    +model=daps_sd_wrapper \
    +sampler=dcdp \
    add_exp_name=dcdp_box
