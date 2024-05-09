from .nodes.flashface_generator import FlashFaceGenerator
from .nodes.flashface_cliptextencode import FlashFaceCLIPTextEncode

NODE_CLASS_MAPPINGS = {
    "FlashFaceGenerator": FlashFaceGenerator,
    "FlashFaceCLIPTextEncode": FlashFaceCLIPTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashFaceGenerator": "📸 FlashFace Generator",
    "FlashFaceCLIPTextEncode": "📸 FlashFace CLIP Text Encode",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'WEB_DIRECTORY']

print("\033[34mFlashFace Nodes: \033[92mLoaded\033[0m")

