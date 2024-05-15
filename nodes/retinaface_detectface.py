import copy

import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import ImageDraw, ImageOps, ImageSequence
from ..ldm.models.retinaface import crop_face
from ..flashface.all_finetune.utils import PadToSquare, get_padding

class RetinaFaceDetectFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "retinaface_model": ("MODEL",),
                "reference_images": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("PIL_IMAGE", "IMAGE")
    RETURN_NAMES = ("pil_images", "cropped_preview")
    INPUT_IS_LIST = (False, True)
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "detect_face"

    def detect_face(self, retinaface_model, reference_images):
        retinaface_model = retinaface_model[0]
        retinaface_transforms = T.Compose([PadToSquare(size=640), T.ToTensor()])

        pil_imgs = []
        # convert each image to PIL and append to list
        for img in reference_images:
            img = img.squeeze(0)
            img = img.permute(2, 0, 1)
            pil_image = F.to_pil_image(img)
            pil_imgs.append(pil_image)

        # read images
        b = len(pil_imgs)
        vis_pil_imgs = copy.deepcopy(pil_imgs)

        # detection
        imgs = torch.stack([retinaface_transforms(u) for u in pil_imgs]).to('cuda')
        boxes, kpts = retinaface_model.detect(imgs, min_thr=0.6)

        # undo padding and scaling
        face_imgs = []

        for i in range(b):
            # params
            scale = 640 / max(pil_imgs[i].size)
            left, top, _, _ = get_padding(round(scale * pil_imgs[i].width),
                                          round(scale * pil_imgs[i].height), 640)

            # undo padding
            boxes[i][:, [0, 2]] -= left
            boxes[i][:, [1, 3]] -= top
            kpts[i][:, :, 0] -= left
            kpts[i][:, :, 1] -= top

            # undo scaling
            boxes[i][:, :4] /= scale
            kpts[i][:, :, :2] /= scale

            # crop faces
            crops = crop_face(pil_imgs[i], boxes[i], kpts[i])
            if len(crops) != 1:
                raise ValueError('Expected 1 face per image, but got %d' % len(crops))

            face_imgs += crops
            # draw boxes on the pil image
            draw = ImageDraw.Draw(vis_pil_imgs[i])
            for box in boxes[i]:
                box = box[:4].tolist()
                box = [int(x) for x in box]
                draw.rectangle(box, outline='red', width=4)

        # convert to tensor for preview
        output_images = []
        for face_img in face_imgs:
            for i in ImageSequence.Iterator(face_img):
                i = ImageOps.exif_transpose(i)
                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                output_images.append(image)

        return ([face_imgs], [output_images],)
