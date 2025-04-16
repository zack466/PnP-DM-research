# TODO: add pnpdm sampler

python posterior_sample.py \
    +data=images_with_prompts \
    +task=super_resolution_svd \
    task.operator.ratio=8 \
    +model=daps_sd_wrapper \
    +sampler=daps \
    sampler.skip_noising=true \
    add_exp_name=pnpdm_superres8 \
    gpu=0

python posterior_sample.py \
    +data=images_with_prompts \
    +task=super_resolution_svd \
    task.operator.ratio=16 \
    +model=daps_sd_wrapper \
    +sampler=daps \
    sampler.skip_noising=true \
    add_exp_name=pnpdm_superres16 \
    gpu=0

python posterior_sample.py \
    +data=images_with_prompts \
    +task=gaussian_deblur_circ \
    task.operator.intensity=3 \
    +model=daps_sd_wrapper \
    +sampler=daps \
    sampler.skip_noising=true \
    sampler.sample_learning_rate=1e-5 \
    add_exp_name=pnpdm_deblur3 \
    gpu=0

python posterior_sample.py \
    +data=images_with_prompts \
    +task=gaussian_deblur_circ \
    task.operator.intensity=6 \
    +model=daps_sd_wrapper \
    +sampler=daps \
    sampler.skip_noising=true \
    sampler.sample_learning_rate=1e-5 \
    add_exp_name=pnpdm_deblur6 \
    gpu=0

python posterior_sample.py \
    +data=images_with_prompts \
    +task=box_inpainting \
    +model=daps_sd_wrapper \
    +sampler=daps \
    sampler.skip_noising=true \
    sampler.sample_learning_rate=2e-6 \
    add_exp_name=pnpdm_box \
    gpu=0
