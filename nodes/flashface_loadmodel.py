import os
import torch

from ..flashface.all_finetune.config import cfg
from ..flashface.all_finetune.ops.context_diffusion import ContextGaussianDiffusion
from ..flashface.all_finetune.models import sd_v1_ref_unet
from ..ldm import sd_v1_vae, ops, data, models


class FlashFaceLoadModel:
    @classmethod
    def INPUT_TYPES(s):
        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Navigate to the parent directory (adjust the number of ".." as needed)
        parent_dir = os.path.join(current_dir, "../../../")

        # Specify the relative path to the "models/flashface" direct
        models_dir = os.path.join(parent_dir, "models/flashface")
        return {
            "required": {
                "flashface_file": (sorted(os.listdir(models_dir)),),
            },
        }
    
    RETURN_TYPES = ("MODEL", "MODEL", "MODEL", "CLIP")
    RETURN_NAMES = ("autoencoder", "flashface model", "diffusion model")
    FUNCTION = "load_models"

    def load_models(self, flashface_file):
        clip_tokenizer = data.CLIPTokenizer(padding='eos')
        clip = getattr(models, cfg.clip_model)(pretrained=True).eval().requires_grad_(False).textual.to('cuda')
        autoencoder = sd_v1_vae(pretrained=True).eval().requires_grad_(False).to('cuda')
        flashface_model = sd_v1_ref_unet(pretrained=True, version='sd-v1-5_nonema', enable_encoder=False).to('cuda')
        flashface_model.replace_input_conv()
        flashface_model = flashface_model.eval().requires_grad_(False).to('cuda')
        flashface_model.share_cache['num_pairs'] = cfg.num_pairs

        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Navigate to the parent directory (adjust the number of ".." as needed)
        parent_dir = os.path.join(current_dir, "../../../")

        # Specify the relative path to the "models/flashface" direct
        models_dir = os.path.join(parent_dir, "models/flashface")

        flashface_full_path = os.path.join(models_dir, flashface_file)
        model_weight = torch.load(flashface_full_path, map_location='cpu')
        msg = flashface_model.load_state_dict(model_weight, strict=True)
        print(msg)
        
        sigmas = ops.noise_schedule(schedule=cfg.schedule, n=cfg.num_timesteps, beta_min=cfg.scale_min, beta_max=cfg.scale_max)
        diffusion = ContextGaussianDiffusion(sigmas=sigmas, prediction_type=cfg.prediction_type)
        diffusion.num_pairs = cfg.num_pairs

        return (autoencoder, flashface_model, diffusion, (clip, clip_tokenizer))