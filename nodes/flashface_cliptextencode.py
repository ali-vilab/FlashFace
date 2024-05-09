class FlashFaceCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": False, "default": ""}),
            },
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "FlashFace"
    
    def encode(self, prompt):
        c = encode_text(clip, clip_tokenizer([prompt]).to(gpu))
        c = c[None].repeat(num_sample, 1, 1, 1).flatten(0, 1)
        c = {'context': c}
        return (c,)
