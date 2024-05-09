from .nodes.flashface_generator import FlashFaceGenerator

NODE_CLASS_MAPPINGS = {
    "FlashFaceGenerator": FlashFaceGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashFaceGenerator": "📸 FlashFace Generator"
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'WEB_DIRECTORY']

print("\033[34mFlashFace Nodes: \033[92mLoaded\033[0m")

