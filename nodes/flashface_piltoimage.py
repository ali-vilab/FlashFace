import numpy as np
import torchvision.transforms.functional as F

class FlashFacePILToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("PIL_IMAGE", {}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "convert"

    def convert(self, images):
        torch_imgs = []
        for img in images:
            img_tensor = F.to_tensor(img)
            # Ensure the data type is correct
            img_np = img_tensor.permute(1, 2, 0)

            torch_imgs.append(img_np)

        return (torch_imgs,)