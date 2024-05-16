import copy

import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import ImageDraw, ImageOps, ImageSequence
from ..ldm.models.retinaface import crop_face
from ..flashface.all_finetune.utils import PadToSquare, get_padding

class FlashFaceImageToPIL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("PIL_IMAGE", )
    RETURN_NAMES = ("cropped_images", )
    INPUT_IS_LIST = (True, )
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "convert"

    def convert(self, reference_images):
        pil_imgs = []
        # convert each image to PIL and append to list
        for img in reference_images:
            img = img.squeeze(0)
            img = img.permute(2, 0, 1)
            pil_image = F.to_pil_image(img)
            pil_imgs.append(pil_image)

        return ([pil_imgs], )
