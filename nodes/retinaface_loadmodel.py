import os

from ..ldm.models.retinaface import retinaface


class RetinaFaceLoadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "retinaface_file": (sorted(os.listdir(get_model_path())),),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("retinaface_model",)
    FUNCTION = "load_retinaface"

    def load_retinaface(self, retinaface_file):
        rf_model = retinaface(pretrained=True, device='cuda', ckpt_path=os.path.join(get_model_path(), retinaface_file)).eval().requires_grad_(False)
        return (rf_model,)


def get_model_path():
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the parent directory (adjust the number of ".." as needed)
    parent_dir = os.path.join(current_dir, "../../../")

    # Specify the relative path to the "models/flashface" direct
    models_dir = os.path.join(parent_dir, "models/retinaface")

    return models_dir
