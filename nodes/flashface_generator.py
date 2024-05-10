class FlashFaceGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "autoencoder": ("MODEL", {}),
                "flashface_model": ("MODEL", {}),
                "diffusion_model": ("MODEL", {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "FlashFace"

    def generate(self, autoencoder, flashface_model, diffusion_model, positive, negative, **kwargs):
        image1 = kwargs.get("image1")

        return (image1,)
