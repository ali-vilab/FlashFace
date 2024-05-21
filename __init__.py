from .nodes.flashface_generator import FlashFaceGenerator
from .nodes.flashface_cliptextencode import FlashFaceCLIPTextEncode
from .nodes.flashface_loadmodel import FlashFaceLoadModel
from .nodes.flashface_detectface import FlashFaceDetectFace

NODE_CLASS_MAPPINGS = {
    "FlashFaceGenerator": FlashFaceGenerator,
    "FlashFaceCLIPTextEncode": FlashFaceCLIPTextEncode,
    "FlashFaceLoadModel": FlashFaceLoadModel,
    "FlashFaceDetectFace": FlashFaceDetectFace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashFaceGenerator": "⚡🎭FlashFace Generator",
    "FlashFaceCLIPTextEncode": "⚡🎭FlashFace CLIP Text Encode",
    "FlashFaceLoadModel": "⚡🎭FlashFace Load Model",
    "FlashFaceDetectFace": "⚡🎭FlashFace Detect Face",

}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'WEB_DIRECTORY']

print("\033[34mFlashFace Nodes: \033[92mLoaded\033[0m")

