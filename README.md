# CS 163 Project Notes

Our codebase is an extension of the original PnP-DM codebase. The main changes we made were adding a new denoiser which uses a preconditioned stable diffusion model and adding a new sampler which modifies the PnP-DM algorithm as described in our report. To run our algorithm for solving inverse problems with text conditioning, set up the repository as normal and use commands of the form:
```
python posterior_sample.py +data=ffhq +task=super_resolution_svd +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm_latent \
       sampler.mode=edm_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=superresolution \
       sampler.text_prompt='a painting in the style of van gogh'
```

# Code for "Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors"

### 1) Install packages
```python
conda create -n pnpdm python=3.10
conda activate pnpdm
pip install -r requirements.txt
```

### 2) Download pretrained checkpoint

Download the corresponding checkpoint from the links below and move it to ```./models/```.
 - [FFHQ (color)](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing)
 - [FFHQ (grayscale)](https://caltech.box.com/s/j58w0bf2pe2t0lrzoq45du0hc55ba4lc)
 - [Blackhole (grayscale)](https://caltech.box.com/s/j58w0bf2pe2t0lrzoq45du0hc55ba4lc)

### 4) Modify the dataset and model paths in config files
You need to modify the paths in the following files so that the dataset and models can be loaded properly:
 - `./configs/data/ffhq.yaml`
 - `./configs/data/ffhq_grayscale.yaml`
 - `./configs/model/edm_unet_adm_blackhole.yaml`
 - `./configs/model/edm_unet_adm_dps_ffhq.yaml`
 - `./configs/model/edm_unet_adm_gray_ffhq.yaml`

### 4) Run experiments
All the commands for running our experiments are provided in ```commands.sh```.
The experiments are configured using [hydra](https://hydra.cc/). 
Please see its documentation for detailed usage.

### 5) Citation
Thank you for your interest in our work!
Please consider citing 
```
@misc{wu2024principled,
      title={Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors}, 
      author={Zihui Wu and Yu Sun and Yifan Chen and Bingliang Zhang and Yisong Yue and Katherine L. Bouman},
      year={2024},
      eprint={2405.18782},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2405.18782}, 
}
```
Please email zwu2@caltech.edu if you run into any problems.
