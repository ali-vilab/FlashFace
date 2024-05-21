import copy
import random

import numpy as np
import torch
import torch.cuda.amp as amp
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import ImageOps, ImageSequence
from PIL import Image, ImageDraw

from ..flashface.all_finetune.config import cfg
from ..flashface.all_finetune.utils import Compose, PadToSquare, seed_everything, get_padding
from ..ldm.models.retinaface import crop_face, retinaface
import comfy.samplers

padding_to_square = PadToSquare(224)

retinaface_transforms = T.Compose([PadToSquare(size=640), T.ToTensor()])

retinaface = retinaface(pretrained=True, device='cuda').eval().requires_grad_(False)

class FlashFaceDetectFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
        }
    RETURN_TYPES = ("PIL_IMAGE", )
    FUNCTION = "detect_face"
    CATEGORY = "FlashFace"

    def detect_face(self, **kwargs):


        imgs = []

        for k, v in kwargs.items():
            imgs.append(v)

        pil_imgs = []
        # convert each image to PIL and append to list
        for img in imgs:
            img = img.squeeze(0)
            img = img.permute(2, 0, 1)
            pil_image = F.to_pil_image(img)
            pil_imgs.append(pil_image)

        # flatten the list
        # pil_imgs = [item for sublist in imgs for item in sublist]

        # read images
        # pil_imgs = imgs
        b = len(pil_imgs)
        vis_pil_imgs = copy.deepcopy(pil_imgs)

        # detection
        imgs = torch.stack([retinaface_transforms(u) for u in pil_imgs]).to('cuda')
        boxes, kpts = retinaface.detect(imgs, min_thr=0.6)

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
                raise (
                    f'Find {len(crops)} faces in the image {i + 1}, please ensure there is only one face in each image'
                )

            face_imgs += crops

            # draw boxes on the pil image
            draw = ImageDraw.Draw(vis_pil_imgs[i])
            for box in boxes[i]:
                box = box[:4].tolist()
                box = [int(x) for x in box]
                draw.rectangle(box, outline='red', width=4)

        face_imgs = face_imgs

        return (face_imgs, )