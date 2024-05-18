from ..ldm import data
class FlashFaceCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": False, "default": ""}),
                "clip": ("CLIP",),
                "base_prompt": (["Do not append", "Append Positive Base Prompt", "Append Negative Base Prompt"], {"default": "Do not append"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "prompt")
    FUNCTION = "encode"
    CATEGORY = "FlashFace"

    def encode(self, prompt, clip, base_prompt):
        def encode_text(m, x):
            # embeddings
            x = m.token_embedding(x) + m.pos_embedding

            # transformer
            for block in m.transformer:
                x = block(x)

            # output
            x = m.norm(x)

            return x
        appended_prompt = ""
        if base_prompt == "Append Positive Base Prompt":
            appended_prompt = ",best quality, masterpiece,ultra-detailed, UHD 4K, photographic"
        elif base_prompt == "Append Negative Base Prompt":
            appended_prompt = ",blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"

        prompt += appended_prompt

        clip_tokenizer = data.CLIPTokenizer(padding='eos')

        c = encode_text(clip, clip_tokenizer([prompt]).to('cuda'))


        return (c, prompt, )
