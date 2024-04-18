import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import ImageDraw


def seed_everything(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_padding(w, h, size):
    if w > h:
        pad = size - h
        return (0, pad // 2, 0, pad - pad // 2)
    elif h > w:
        pad = size - w
        return (pad // 2, 0, pad - pad // 2, 0)
    else:
        return (0, 0, 0, 0)


class PadToSquare(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        scale = self.size / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)))
        return TF.pad(img, get_padding(img.width, img.height, self.size))


class Compose(object):
    """Sliceable version of T.Compose."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.transforms[index]
        elif isinstance(index, slice):
            return Compose(self.transforms[index])
        else:
            raise TypeError(f'Invalid argument {index}')


def bbox_crop(img, face_box, base_size=(512, 512), vis=False):
    """Crop the square region with max area that contains the face bounding
    box."""

    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Calculate the center point of the face bounding box
    x1, y1, x2, y2 = face_box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    width = img.width
    height = img.height

    def solve_delta(max_delta, value):
        if value < 0:
            return 0
        elif value < max_delta:
            return value
        else:
            return max_delta

    if width > height:
        max_size = height
        delta = (width - height)
        delta = solve_delta(delta, center_x - max_size / 2)
        crop_region = (delta, 0, max_size + delta, max_size)
        new_face_box = (x1 - delta, y1, x2 - delta, y2)
        # clip the new face box in the cropped image
        new_face_box = [max(0, x) for x in new_face_box]
        new_face_box = [min(max_size, x) for x in new_face_box]

    else:
        max_size = width
        delta = (height - width)
        delta = solve_delta(delta, center_y - max_size / 2)
        crop_region = (0, delta, max_size, max_size + delta)
        new_face_box = (x1, y1 - delta, x2, y2 - delta)
        # clip the new face box in the cropped image
        new_face_box = [max(0, x) for x in new_face_box]
        new_face_box = [min(max_size, x) for x in new_face_box]

    # Crop the square region from the image
    cropped_img = img.crop(crop_region)
    # resize the cropped image to base_size
    cropped_img = cropped_img.resize(base_size)

    # draw the new face box on the cropped image

    if vis:
        draw = ImageDraw.Draw(cropped_img)
        draw.rectangle(new_face_box, outline='red')

    return cropped_img, new_face_box
