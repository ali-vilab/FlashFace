from ..ldm import data
class FlashFaceCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": False, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": False, "default": ""}),
                "clip": ("CLIP",),
                "num_sample": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "encode"
    CATEGORY = "FlashFace"

    def encode(self, positive_prompt, negative_prompt, clip, num_sample):
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
        c = c[None].repeat(num_sample, 1, 1, 1).flatten(0, 1)
        c = {'context': c}

        single_null_context = encode_text(clip,
                                          clip_tokenizer([negative_prompt]).cuda()).to('cuda')
        null_context = single_null_context
        nc = {
            'context': null_context[None].repeat(num_sample, 1, 1, 1).flatten(0, 1),
            'snc': single_null_context,
        }

        return (c, nc, )
