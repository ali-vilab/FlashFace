from ..ldm import data
class FlashFaceCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": False, "default": ""}),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "FlashFace"

    def encode(self, prompt, clip):
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

        c = encode_text(clip, clip_tokenizer([prompt]).to('cuda'))


        return (c, )
