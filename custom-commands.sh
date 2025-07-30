for i in {1..10}; do
python posterior_sample.py \
    +data=images_with_prompts \
    +task=super_resolution_svd \
    task.operator.ratio=16 \
    +model=daps_sd_wrapper \
    +sampler=daps \
    gpu=0 \
    add_exp_name="test_6_30_num_$i"
done
