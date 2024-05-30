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

class FlashFaceGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "reference_faces": ("PIL_IMAGE", {}),
                "vae": ("VAE", {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "sampler": (['ddim', 'euler',], ),
                "steps": ("INT", {"default": 35}),
                "text_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "reference_feature_strength": ("FLOAT", {"default": 1.2, "min": 0.7, "max": 1.4, "step": 0.05}),
                "reference_guidance_strength": ("FLOAT", {"default": 3.2, "min": 1.8, "max": 4.0, "step": 0.1}),
                "step_to_launch_face_guidance": ("INT", {"default": 750, "min": 0, "max": 1000, "step": 50}),
                "face_bbox_x1": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "face_bbox_y1": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1}),
                "face_bbox_x2": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "face_bbox_y2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "num_samples": ("INT", {"default": 1}),

            }
        }
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "generate"
    CATEGORY = "FlashFace"


    def generate(self, model, positive, negative, reference_faces, vae, seed, sampler, steps, text_guidance_strength,
                 reference_feature_strength, reference_guidance_strength, step_to_launch_face_guidance, face_bbox_x1,
                 face_bbox_y1, face_bbox_x2, face_bbox_y2, num_samples):

        # reference_faces = [image1[0], image2[0], image3[0], image4[0]]
        seed_everything(seed)

        # reference_faces = detect_face(reference_faces)

        # for i, ref_img in enumerate(reference_faces):
        #     ref_img.save(f'./{i + 1}.png')
        print(f'detected {len(reference_faces)} faces')
        if len(reference_faces) == 0:
            raise (
                'No face detected in the reference images, please upload images with clear face'
            )

        face_transforms = Compose(
            [T.ToTensor(),
             T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        lamda_feat_before_ref_guidence = 0.85

        # process the ref_imgs
        face_bbox = [face_bbox_x1, face_bbox_y1, face_bbox_x2, face_bbox_y2]
        H = W = 768
        if isinstance(face_bbox, str):
            face_bbox = eval(face_bbox)
        normalized_bbox = face_bbox
        face_bbox = [
            int(normalized_bbox[0] * W),
            int(normalized_bbox[1] * H),
            int(normalized_bbox[2] * W),
            int(normalized_bbox[3] * H)
        ]
        max_size = max(face_bbox[2] - face_bbox[1], face_bbox[3] - face_bbox[1])
        empty_mask = torch.zeros((H, W))

        empty_mask[face_bbox[1]:face_bbox[1] + max_size,
        face_bbox[0]:face_bbox[0] + max_size] = 1

        empty_mask = empty_mask[::8, ::8].cuda()
        empty_mask = empty_mask[None].repeat(num_samples, 1, 1)

        padding_to_square = PadToSquare(224)
        pasted_ref_faces = []
        show_refs = []
        for ref_img in reference_faces:
            ref_img = ref_img.convert('RGB')
            ref_img = padding_to_square(ref_img)
            to_paste = ref_img

            to_paste = face_transforms(to_paste)
            pasted_ref_faces.append(to_paste)

        faces = torch.stack(pasted_ref_faces, dim=0).to('cuda')

        ref_z0 = cfg.ae_scale * torch.cat([
            vae.sample(u, deterministic=True)
            for u in faces.split(cfg.ae_batch_size)
        ])
        model, diffusion = model
        model.share_cache['num_pairs'] = len(faces)
        model.share_cache['ref'] = ref_z0
        model.share_cache['similarity'] = torch.tensor(reference_feature_strength).cuda()
        model.share_cache['ori_similarity'] = torch.tensor(reference_feature_strength).cuda()
        model.share_cache['lamda_feat_before_ref_guidance'] = torch.tensor(lamda_feat_before_ref_guidence).cuda()
        model.share_cache['ref_context'] = negative.repeat(len(ref_z0), 1, 1)
        model.share_cache['masks'] = empty_mask
        model.share_cache['classifier'] = reference_guidance_strength
        model.share_cache['step_to_launch_face_guidance'] = step_to_launch_face_guidance

        diffusion.classifier = reference_guidance_strength

        progress = 0.0
        diffusion.progress = 0

        positive = positive[None].repeat(num_samples, 1, 1, 1).flatten(0, 1)
        positive = {'context': positive}

        negative = {
            'context': negative[None].repeat(num_samples, 1, 1, 1).flatten(0, 1)
        }
        # sample
        with amp.autocast(dtype=cfg.flash_dtype), torch.no_grad():
            z0 = diffusion.sample(solver=sampler,
                                  noise=torch.empty(num_samples,
                                                    4,
                                                    768 // 8,
                                                    768 // 8,
                                                    device='cuda').normal_(),
                                  model=model,
                                  model_kwargs=[positive, negative],
                                  steps=steps,
                                  guide_scale=text_guidance_strength,
                                  guide_rescale=0.5,
                                  show_progress=True,
                                  discretization=cfg.discretization)

        imgs = vae.decode(z0 / cfg.ae_scale)
        del model.share_cache['ori_similarity']
        # output
        imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(
            0, 255).astype(np.uint8)

        # convert to PIL image
        imgs_pil = [Image.fromarray(img) for img in imgs]
        imgs_pil = imgs_pil + show_refs

        # save imgs to file
        # for i, img in enumerate(imgs):
        #     img.save(f"sample_{i}.png")

        torch_imgs = []
        for img in imgs_pil:
            img_tensor = F.to_tensor(img)
            # Ensure the data type is correct
            img_np = img_tensor.permute(1, 2, 0)

            torch_imgs.append(img_np)

        return (torch_imgs, )