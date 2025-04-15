python posterior_sample.py \
    +data=images_with_prompts \
    +task=super_resolution_svd \
    +model=daps_sd_wrapper \
    +sampler=daps \
    add_exp_name=daps_superres

python posterior_sample.py \
    +data=images_with_prompts \
    +task=gaussian_deblur_circ \
    +model=daps_sd_wrapper \
    +sampler=daps \
    add_exp_name=daps_deblur

python posterior_sample.py \
    +data=images_with_prompts \
    +task=box_inpainting \
    +model=daps_sd_wrapper \
    +sampler=daps \
    add_exp_name=daps_box
