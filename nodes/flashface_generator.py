import random
from PIL import Image

class FlashFaceGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
        }
    }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "FlashFace"

    def generate(self, **kwargs):
        image1 = kwargs.get("image1")

        return (image1, )