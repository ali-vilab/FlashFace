from ..ldm import data
class FlashFaceCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": False, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": False, "default": ""}),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "encode"
    CATEGORY = "FlashFace"

    def encode(self, positive_prompt, negative_prompt, clip):
        def encode_text(m, x):
            # embeddings
            x = m.token_embedding(x) + m.pos_embedding

            # transformer
            for block in m.transformer:
                x = block(x)

            # output
            x = m.norm(x)

            return x

        clip_tokenizer = data.CLIPTokenizer(padding='eos')

        c = encode_text(clip, clip_tokenizer([positive_prompt]).to('cuda'))
        nc = encode_text(clip, clip_tokenizer([negative_prompt]).cuda()).to('cuda')


        return (c, nc, )
