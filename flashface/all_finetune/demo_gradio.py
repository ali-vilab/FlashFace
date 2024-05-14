import copy
import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gradio as gr
import numpy as np
import torch
import torch.cuda.amp as amp
import torchvision.transforms as T
from config import cfg
from models import sd_v1_ref_unet
from ops.context_diffusion import ContextGaussianDiffusion
from PIL import Image, ImageDraw
from utils import Compose, PadToSquare, get_padding, seed_everything

from ldm import data, models, ops
from ldm.models.retinaface import crop_face, retinaface
from ldm.models.vae import sd_v1_vae

# model path
SKIP_LOAD = False
DEBUG_VIEW = False
SKEP_LOAD = False
LOAD_FLAG = True
DEFAULT_INPUT_IMAGES = 4
MAX_INPUT_IMAGES = 4
SIZE = 768
with_lora = False
enable_encoder = False

weight_path = './cache/flashface.ckpt'

gpu = 'cuda'

padding_to_square = PadToSquare(224)

retinaface_transforms = T.Compose([PadToSquare(size=640), T.ToTensor()])

retinaface = retinaface(pretrained=True,
                        device='cuda').eval().requires_grad_(False)


def detect_face(imgs=None):

    # read images
    pil_imgs = imgs
    b = len(pil_imgs)
    vis_pil_imgs = copy.deepcopy(pil_imgs)

    # detection
    imgs = torch.stack([retinaface_transforms(u) for u in pil_imgs]).to(gpu)
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
            raise gr.Error(
                f'Find {len(crops)} faces in the image {i+1}, please ensure there is only one face in each image'
            )

        face_imgs += crops

        # draw boxes on the pil image
        draw = ImageDraw.Draw(vis_pil_imgs[i])
        for box in boxes[i]:
            box = box[:4].tolist()
            box = [int(x) for x in box]
            draw.rectangle(box, outline='red', width=4)

    face_imgs = face_imgs

    return face_imgs


if not DEBUG_VIEW and not SKEP_LOAD:
    clip_tokenizer = data.CLIPTokenizer(padding='eos')
    clip = getattr(models, cfg.clip_model)(
        pretrained=True).eval().requires_grad_(False).textual.to(gpu)
    autoencoder = sd_v1_vae(
        pretrained=True).eval().requires_grad_(False).to(gpu)

    unet = sd_v1_ref_unet(pretrained=True,
                          version='sd-v1-5_nonema',
                          enable_encoder=enable_encoder).to(gpu)

    unet.replace_input_conv()
    unet = unet.eval().requires_grad_(False).to(gpu)
    unet.share_cache['num_pairs'] = cfg.num_pairs

    if LOAD_FLAG:
        model_weight = torch.load(weight_path, map_location='cpu')
        msg = unet.load_state_dict(model_weight, strict=True)
        print(msg)

    # diffusion
    sigmas = ops.noise_schedule(schedule=cfg.schedule,
                                n=cfg.num_timesteps,
                                beta_min=cfg.scale_min,
                                beta_max=cfg.scale_max)
    diffusion = ContextGaussianDiffusion(sigmas=sigmas,
                                         prediction_type=cfg.prediction_type)
    diffusion.num_pairs = cfg.num_pairs

face_transforms = Compose(
    [T.ToTensor(),
     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def encode_text(m, x):
    # embeddings
    x = m.token_embedding(x) + m.pos_embedding

    # transformer
    for block in m.transformer:
        x = block(x)

    # output
    x = m.norm(x)

    return x


def generate(
    pos_prompt,
    neg_prompt,
    steps=30,
    face_bbox=[0.3, 0.1, 0.6, 0.4],
    lamda_feat=1.2,
    face_guidence=3.2,
    num_sample=1,
    text_control_scale=7.5,
    seed=-1,
    step_to_launch_face_guidence=750,
    reference_face_1=None,
    reference_face_2=None,
    reference_face_3=None,
    reference_face_4=None,
    default_pos_prompt='best quality, masterpiece,ultra-detailed, UHD 4K, photographic',
    default_neg_prompt='blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face',
    need_detect=True,
    lamda_feat_before_ref_guidence=0.85,
    progress=gr.Progress()):
    reference_faces = [
        reference_face_1, reference_face_2, reference_face_3, reference_face_4
    ]
    # filter none
    reference_faces = [ref for ref in reference_faces if ref is not None]
    solver = 'ddim'
    if default_pos_prompt is not None:
        pos_prompt = pos_prompt + ', ' + default_pos_prompt
    if neg_prompt is not None and len(neg_prompt) > 0:
        neg_prompt = neg_prompt + ', ' + default_neg_prompt
    else:
        neg_prompt = default_neg_prompt
    if seed == -1:
        seed = random.randint(0, 2147483647)
    print(seed)
    seed_everything(seed)
    print('final pos_prompt: ', pos_prompt)
    print('final neg_prompt: ', neg_prompt)
    progress(0, desc='Face detection')

    if need_detect:
        reference_faces = detect_face(reference_faces)

        # for i, ref_img in enumerate(reference_faces):
        #     ref_img.save(f'./{i + 1}.png')
        print(f'detected {len(reference_faces)} faces')
        if len(reference_faces) == 0:
            raise gr.Error(
                'No face detected in the reference images, please upload images with clear face'
            )

        if len(reference_faces) < 4:
            expand_reference_faces = copy.deepcopy(reference_faces)
            while len(expand_reference_faces) < 4:
                # random select from ref_imgs
                expand_reference_faces.append(random.choice(reference_faces))
            reference_faces = expand_reference_faces

    # process the ref_imgs
    H = W = 768
    if isinstance(face_bbox, str):
        face_bbox = eval(face_bbox)
    normalized_bbox = face_bbox
    print(normalized_bbox)
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
    empty_mask = empty_mask[None].repeat(num_sample, 1, 1)

    pasted_ref_faces = []
    show_refs = []
    for ref_img in reference_faces:
        ref_img = ref_img.convert('RGB')
        ref_img = padding_to_square(ref_img)
        to_paste = ref_img

        to_paste = face_transforms(to_paste)
        pasted_ref_faces.append(to_paste)

    faces = torch.stack(pasted_ref_faces, dim=0).to(gpu)

    c = encode_text(clip, clip_tokenizer([pos_prompt]).to(gpu))
    c = c[None].repeat(num_sample, 1, 1, 1).flatten(0, 1)
    c = {'context': c}

    single_null_context = encode_text(clip,
                                      clip_tokenizer([neg_prompt
                                                      ]).cuda()).to(gpu)
    null_context = single_null_context
    nc = {
        'context': null_context[None].repeat(num_sample, 1, 1,
                                             1).flatten(0, 1)
    }

    ref_z0 = cfg.ae_scale * torch.cat([
        autoencoder.sample(u, deterministic=True)
        for u in faces.split(cfg.ae_batch_size)
    ])
    #  ref_z0 = ref_z0[None].repeat(num_sample, 1,1,1,1).flatten(0,1)
    unet.share_cache['num_pairs'] = 4
    unet.share_cache['ref'] = ref_z0
    unet.share_cache['similarity'] = torch.tensor(lamda_feat).cuda()
    unet.share_cache['ori_similarity'] = torch.tensor(lamda_feat).cuda()
    unet.share_cache['lamda_feat_before_ref_guidence'] = torch.tensor(
        lamda_feat_before_ref_guidence).cuda()
    unet.share_cache['ref_context'] = single_null_context.repeat(
        len(ref_z0), 1, 1)
    unet.share_cache['masks'] = empty_mask
    unet.share_cache['classifier'] = face_guidence
    unet.share_cache[
        'step_to_launch_face_guidence'] = step_to_launch_face_guidence

    diffusion.classifier = face_guidence

    progress(0, desc='starting')
    diffusion.progress = progress
    # sample
    with amp.autocast(dtype=cfg.flash_dtype), torch.no_grad():
        z0 = diffusion.sample(solver=solver,
                              noise=torch.empty(num_sample,
                                                4,
                                                768 // 8,
                                                768 // 8,
                                                device=gpu).normal_(),
                              model=unet,
                              model_kwargs=[c, nc],
                              steps=steps,
                              guide_scale=text_control_scale,
                              guide_rescale=0.5,
                              show_progress=True,
                              discretization=cfg.discretization)

    imgs = autoencoder.decode(z0 / cfg.ae_scale)
    del unet.share_cache['ori_similarity']
    # output
    imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(
        0, 255).astype(np.uint8)

    # convert to PIL image
    imgs = [Image.fromarray(img) for img in imgs]
    imgs = imgs + show_refs

    return imgs

block = gr.Blocks().queue()
with block:
    gr.Markdown(
            '# <center> FlashFace-SD1.5 </center>'
        )
    with gr.Column():
        gr.Markdown(
            '请上传 1 到 4 张包含包含您脸部的参考图像，**提供的人脸图片越多样，生成的效果越好**, 请确保一张图只有一个人脸，如包含多张人脸或者不清晰人脸可能会导致应用无法正常运行'
        )
        gr.Markdown(
            'Please upload 1 to 4 reference images containing only your face. **The greater the diversity of provided facial images, the better the generated results will be.**. please make sure there is only one face in each image. If there are multiple faces or unclear faces in the image, the application may not work properly.'
        )
        with gr.Row():

            reference_images = []
            for i in range(MAX_INPUT_IMAGES):
                reference_images.append(
                    gr.Image(label=f'Reference Image {i+1}',
                             type='pil',
                             visible=True,
                             value=None))

    with gr.Row():
        with gr.Column():

            run_button = gr.Button(label='Run', value='Run')
            num_sample = gr.Slider(label='Num Generated Images',
                                   minimum=0,
                                   maximum=10,
                                   value=4,
                                   step=1)
            steps = gr.Slider(label='Num Steps',
                              minimum=0,
                              maximum=50,
                              value=35,
                              step=1)
            seed = gr.Slider(label='Seed',
                             minimum=-1,
                             maximum=2147483647,
                             step=1,
                             value=-1)
            gr.Markdown(
                '``face position`` 是归一化后的人脸坐标，格式为 [x1,y1,x2,y2],例如 [0.1,0.1,0.5,0.5]表示人脸位置在图片的左上角，[0.,0.,0.,0.]则表示不约束人脸位置,'
            )
            gr.Markdown(
                "``face position`` with the format of normalized coordinates [x1,y1,x2,y2]', For example, [0.1,0.1,0.5,0.5] means the face position is in the upper left corner of the image, [0.,0.,0.,0.] means do not constrain the face position"
            )
            bbox_text = gr.Textbox(label='face position ',
                                   value='[0.3, 0.1, 0.6, 0.4]')

        with gr.Column():
            gr.Markdown(
                '包含 **handsome young man** 或者 **beautiful young woman** 在 ``Positive Prompt`` 可以让您的脸部更加吸引人'
            )
            gr.Markdown(
                'Including **handsome young man** or **beautiful young woman** can make your face more attractive'
            )
            pos_prompt = gr.Textbox(
                label='Positive Prompt',
                value=
                'A handsome young man , with long brown hair, on the street in the night, soft light'
            )
            gr.Markdown(
                '您可以使用 ``Negative Prompt`` 来去除生成图像中一些不需要的视觉元素，比如**胡子，眼镜，刘海 等**')
            gr.Markdown(
                'You can use ``Negative Prompt`` to remove some unwanted features in the generated image, such as **beard, glasses, bangs, etc.**'
            )
            neg_prompt = gr.Textbox(label='Negative Prompt', value='')

            text_control_scale = gr.Slider(label='Text Guidance Strength',
                                           minimum=0,
                                           maximum=10.0,
                                           value=7.5,
                                           step=0.1)

            default_pos_prompt = gr.Textbox(
                label='Default Positive Prompt Postfix',
                value=
                'best quality, masterpiece,ultra-detailed, UHD 4K, photographic'
            )

            default_neg_prompt = gr.Textbox(
                label=' Default Negative Prompt Postfix',
                value=
                'blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face'
            )

        with gr.Column():
            gr.Markdown(
                '使用更大的 ``Reference Feature Strength``，  ``Reference Guidance Strength`` 和 ``Step Index to Launch Ref Guidance`` 可以让生成人脸图像更加忠实于参考图像中的人脸ID，但也可能导致生成的图像中人脸多样性降低. <br><br> **一般情况,固定其他参数为默认，仅仅调大``Reference Feature Strength``一个参数就足够**， 除非你想保留非常多面部细节'
            )
            gr.Markdown(
                'Bigger ``Reference Feature Strength``，  ``Reference Guidance Strength`` and ``Step Index to Launch Ref Guidance`` will lead to more fidelity to the face identity in the reference images, but may also cause less face diversity in the generated images. <br><br> **Generally, keep other parameters at default, and just increase ``Reference Feature Strength`` is enough**, unless you want to retain a lot of facial details'
            )
            similarity = gr.Slider(label='Reference Feature Strength',
                                   minimum=0.7,
                                   maximum=1.4,
                                   value=1.2,
                                   step=0.05)
            classifier = gr.Slider(label='Reference Guidance Strength',
                                   minimum=1.8,
                                   maximum=4,
                                   value=3.2,
                                   step=0.1)
            step_to_launch_face_guidence = gr.Slider(
                label='Step Index to Launch Ref Guidance',
                minimum=0,
                maximum=1000,
                value=750,
                step=50)

    with gr.Row():

        result_gallery = gr.Gallery(label='Output',
                                    show_label=True,
                                    width=SIZE // 2,
                                    height=SIZE // 2,
                                    elem_id='gallery').style(grid=4,
                                                             height='auto')
    with gr.Row():
        gr.Examples(examples=[
            [
                'A handsome young man with long brown hair is sitting in the desert',
                'examples/avatar.png', None, None, None, 1, 2.5, 700, 2, 0,
                25, '[0.3, 0.1, 0.6, 0.4]', 7.5, ''
            ],
 
            [
                'An handsome young man, with cowboy hat, long hair, full body, standing in the forest, sunset',
                'examples/man_face/1.png', 'examples/man_face/2.png',
                'examples/man_face/3.png', 'examples/man_face/4.png', 1.2, 3.2,
                750, 2, 0, 25, '[0.3, 0.1, 0.6, 0.4]', 7.5, 'glasses, beard'
            ],
            
            [
                'A very old man, with short wavy hair',
                'examples/man_face/1.png', 'examples/man_face/2.png',
                'examples/man_face/3.png', 'examples/man_face/4.png', 0.9, 2,
                750, 2, 0, 25, '[0.3, 0.1, 0.6, 0.4]', 7.5, ''
            ],

        ],
                    inputs=[
                        pos_prompt,
                        reference_images[0],
                        reference_images[1],
                        reference_images[2],
                        reference_images[3],
                        similarity,
                        classifier,
                        step_to_launch_face_guidence,
                        num_sample,
                        seed,
                        steps,
                        bbox_text,
                        text_control_scale,
                        neg_prompt,
                    ],
                    cache_examples=False,
                    examples_per_page=100)

    run_button.click(fn=generate,
                     inputs=[
                         pos_prompt, neg_prompt, steps, bbox_text, similarity,
                         classifier, num_sample, text_control_scale, seed,
                         step_to_launch_face_guidence, *reference_images,
                         default_pos_prompt, default_neg_prompt
                     ],
                     outputs=[result_gallery])

block.launch(server_name='0.0.0.0')