class FlashFaceGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positiive": ("CONDITIONING", {}),
            "negative": ("CONDITIONING", {}),
    
        }
    }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "FlashFace"

    def generate(self, positive, negative, **kwargs):
        image1 = kwargs.get("image1")

        return (image1, )