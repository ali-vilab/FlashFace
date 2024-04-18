import os.path as osp
import sys

import torch
from easydict import EasyDict

cfg = EasyDict(__name__='Config: Text-to-Image Model')

sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))

cfg.num_pairs = 4

# diffusion
cfg.schedule = 'scaled_linear'
cfg.num_timesteps = 1000
cfg.zero_terminal_snr = False
cfg.scale_min = 0.00085
cfg.scale_max = 0.0120
cfg.prediction_type = 'eps'

# sampling
cfg.solver = 'heun'
cfg.sampling_steps = 50
cfg.guide_scale = 5.0
cfg.guide_rescale = 0.5
cfg.discretization = 'trailing'

# ------------------------ model ------------------------#

# clip
cfg.clip_model = 'clip_vit_l_14'
cfg.text_len = 77

# autoencoder
cfg.ae_model = 'sd_v1_vae'
cfg.ae_scale = 0.18215
cfg.ae_batch_size = 3

# unet
cfg.num_heads = 8
cfg.flash_dtype = torch.float16
cfg.freeze_backbone = False
cfg.misc_dropout = 0.5
cfg.p_all_zero = 0.1
cfg.p_all_keep = 0.1
