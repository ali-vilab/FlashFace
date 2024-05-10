class FlashFaceGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "autoencoder": ("MODEL", {}),
            "flashface model": ("MODEL", {}),
            "diffusion model": ("MODEL", {}),
            "positiive": ("CONDITIONING", {}),
            "negative": ("CONDITIONING", {}),
    
        }
    }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "FlashFace"

    def generate(self, autoencoder, flashface_model, diffusion_model, positive, negative, **kwargs):
        image1 = kwargs.get("image1")

        return (image1, )