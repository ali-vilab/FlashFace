"""Concise re-implementation of ``https://github.com/serengil/retinaface'' and
``https://github.com/biubug6/Pytorch_Retinaface''."""
import itertools
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

__all__ = ['RetinaFace', 'retinaface', 'crop_face']

CONFIGS = {
    'mobilenetv1': {
        'layers': {
            'stage1': 1,
            'stage2': 2,
            'stage3': 3
        },
        'in_dims': [64, 128, 256],
        'out_dim': 64,
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'strides': [8, 16, 32],
        'mean': [104, 117, 123],
        'variance': [0.1, 0.2]
    },
    'resnet50': {
        'layers': {
            'layer2': 1,
            'layer3': 2,
            'layer4': 3
        },
        'in_dims': [512, 1024, 2048],
        'out_dim': 256,
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'strides': [8, 16, 32],
        'mean': [104, 117, 123],
        'variance': [0.1, 0.2]
    }
}


def conv_bn(in_dim, out_dim, stride=1, leaky=0):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(out_dim),
                         nn.LeakyReLU(negative_slope=leaky, inplace=True))


def conv_bn_no_relu(in_dim, out_dim, stride):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(out_dim))


def conv_bn1X1(in_dim, out_dim, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(negative_slope=leaky, inplace=True))


def conv_dw(in_dim, out_dim, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(in_dim, in_dim, 3, stride, 1, groups=in_dim, bias=False),
        nn.BatchNorm2d(in_dim), nn.LeakyReLU(negative_slope=leaky,
                                             inplace=True),
        nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(negative_slope=leaky, inplace=True))


def prior_box(height, width, min_sizes, strides):
    fpn_sizes = [[
        int(math.ceil(height / stride)),
        int(math.ceil(width / stride))
    ] for stride in strides]
    anchors = []
    for k, size in enumerate(fpn_sizes):
        for i, j in itertools.product(range(size[0]), range(size[1])):
            for min_size in min_sizes[k]:
                sx = min_size / width
                sy = min_size / height
                cx = [x * strides[k] / width for x in [j + 0.5]]
                cy = [y * strides[k] / height for y in [i + 0.5]]
                for cy, cx in itertools.product(cy, cx):
                    anchors += [cx, cy, sx, sy]

    # convert to tensor
    return torch.Tensor(anchors).view(-1, 4)


def nms(dets, thr):
    """Pure Python NMS implementation."""
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort().flip(0)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1 + 1).clamp_(0)
        h = (yy2 - yy1 + 1).clamp_(0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = torch.where(ovr <= thr)[0]
        order = order[inds + 1]
    return torch.LongTensor(keep).to(dets.device)


class SSH(nn.Module):

    def __init__(self, in_dim, out_dim):
        assert out_dim % 4 == 0
        leaky = 0.1 if out_dim <= 64 else 0.0
        super().__init__()
        self.conv3X3 = conv_bn_no_relu(in_dim, out_dim // 2, stride=1)
        self.conv5X5_1 = conv_bn(in_dim, out_dim // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_dim // 4, out_dim // 4, stride=1)
        self.conv7X7_2 = conv_bn(out_dim // 4,
                                 out_dim // 4,
                                 stride=1,
                                 leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_dim // 4, out_dim // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):

    def __init__(self, in_dims, out_dim):
        leaky = 0.1 if out_dim <= 64 else 0.0
        super().__init__()
        self.output1 = conv_bn1X1(in_dims[0], out_dim, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_dims[1], out_dim, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_dims[2], out_dim, stride=1, leaky=leaky)
        self.merge1 = conv_bn(out_dim, out_dim, leaky=leaky)
        self.merge2 = conv_bn(out_dim, out_dim, leaky=leaky)

    def forward(self, input):
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3,
                            size=[output2.size(2),
                                  output2.size(3)],
                            mode='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2,
                            size=[output1.size(2),
                                  output1.size(3)],
                            mode='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class MobileNetV1(nn.Module):

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 + 32 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class ClassHead(nn.Module):

    def __init__(self, in_dim=512, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(in_dim, self.num_anchors * 2, kernel_size=1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):

    def __init__(self, in_dim=512, num_anchors=3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_dim, num_anchors * 4, kernel_size=1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):

    def __init__(self, in_dim=512, num_anchors=3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_dim, num_anchors * 10, kernel_size=1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):

    def __init__(self, backbone='resnet50'):
        assert backbone in CONFIGS
        super().__init__()
        self.cfg = CONFIGS[backbone]

        # body, fpn, ssh
        backbone = MobileNetV1() if backbone == 'mobilenetv1' \
            else resnet50(pretrained=False)
        in_dims, out_dim = self.cfg['in_dims'], self.cfg['out_dim']
        self.body = IntermediateLayerGetter(backbone, self.cfg['layers'])
        self.fpn = FPN(in_dims, out_dim)
        self.ssh1 = SSH(out_dim, out_dim)
        self.ssh2 = SSH(out_dim, out_dim)
        self.ssh3 = SSH(out_dim, out_dim)

        # heads
        self.class_head = nn.ModuleList(
            [ClassHead(out_dim, 2) for _ in range(3)])
        self.bbox_head = nn.ModuleList(
            [BboxHead(out_dim, 2) for _ in range(3)])
        self.landmark_head = nn.ModuleList(
            [LandmarkHead(out_dim, 2) for _ in range(3)])

    def forward(self, imgs):
        # body, fpn, ssh
        fpn = self.fpn(self.body(imgs))
        ssh = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]

        # heads
        scores = torch.cat([self.class_head[i](u) for i, u in enumerate(ssh)],
                           dim=1)
        bboxes = torch.cat([self.bbox_head[i](u) for i, u in enumerate(ssh)],
                           dim=1)
        landmarks = torch.cat(
            [self.landmark_head[i](u) for i, u in enumerate(ssh)], dim=1)
        return scores, bboxes, landmarks

    @torch.no_grad()
    def detect(self,
               imgs,
               mode='RGB',
               min_thr=0.02,
               nms_thr=0.4,
               pre_nms_topk=5000,
               post_nms_topk=750):
        """
        imgs: [B, C, H, W] within range [0, 1].
        """
        # preprocess
        assert mode in ['RGB', 'BGR']
        if mode == 'RGB':
            imgs = imgs.flip(1)
        imgs = 255.0 * imgs - imgs.new_tensor(self.cfg['mean']).view(
            1, -1, 1, 1)

        # params
        b, c, h, w = imgs.size()
        var = self.cfg['variance']

        # forward pass
        scores, bboxes, landmarks = self.forward(imgs)
        scores = F.softmax(scores, dim=-1)[..., 1]
        pre_nms_topk = min(pre_nms_topk, scores.size(1))

        # decode bboxes
        priors = prior_box(h, w, self.cfg['min_sizes'],
                           self.cfg['strides']).unsqueeze(0).to(imgs)
        bboxes = torch.cat([
            priors[..., :2] + bboxes[..., :2] * var[0] * priors[..., 2:],
            priors[..., 2:] * torch.exp(bboxes[..., 2:] * var[1])
        ],
                           dim=-1)
        bboxes[..., :2] -= bboxes[..., 2:] / 2.0
        bboxes[..., 2:] += bboxes[..., :2]
        bboxes *= torch.tensor([w, h, w, h]).to(bboxes)

        # decode landmarks
        landmarks = torch.cat(
            [
                # left-eye
                priors[..., :2] +
                landmarks[..., 0:2] * var[0] * priors[..., 2:],
                # right-eye
                priors[..., :2] +
                landmarks[..., 2:4] * var[0] * priors[..., 2:],
                # nose
                priors[..., :2] +
                landmarks[..., 4:6] * var[0] * priors[..., 2:],
                # left-corner of the mouth
                priors[..., :2] +
                landmarks[..., 6:8] * var[0] * priors[..., 2:],
                # right-corner of the mouth
                priors[..., :2] +
                landmarks[..., 8:10] * var[0] * priors[..., 2:]
            ],
            dim=-1)
        landmarks *= torch.tensor([w, h, w, h, w, h, w, h, w, h]).to(landmarks)

        # loop over images
        out_bboxes = []
        out_landmarks = []
        for i in range(b):
            # pre-nms filtering
            topk_scores, topk_indices = scores[i].topk(pre_nms_topk)
            indices = topk_indices[topk_scores > min_thr]
            preds = torch.cat(
                [bboxes[i, indices], scores[i, indices].unsqueeze(-1)], dim=1)
            lands = landmarks[i, indices]

            # nms & post-nms filtering
            keep = nms(preds, nms_thr)[:post_nms_topk]
            out_bboxes.append(preds[keep])
            out_landmarks.append(lands[keep].view(-1, 5, 2))
        return out_bboxes, out_landmarks


def retinaface(pretrained=False, device='cpu', backbone='resnet50'):
    # init model
    if pretrained:

        # init a model on device
        with torch.device(device):
            model = RetinaFace(backbone=backbone)
        ckpt_path = Path(__file__).parents[2] / "cache" / f"retinaface_{backbone}.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        # init a model on device
        with torch.device(device):
            model = RetinaFace(backbone=backbone)
    return model


def crop_face(img, bboxes, landmarks):
    """
    img: PIL.Image.
    bboxes: torch.FloatTensor of shape [N, 4].
    landmarks: torch.FloatTensor of shape [N, 5, 2].
    """
    crops = []
    for bbox, landmark in zip(bboxes[:, :4], landmarks):
        crop = img.crop(tuple(bbox.round().tolist()))
        leye, reye, nose, _, _ = landmark

        if leye[1] > reye[1]:
            point_3rd = torch.tensor([reye[0], leye[1]], device=bboxes.device)
            direction = -1
        else:
            point_3rd = torch.tensor([leye[0], reye[1]], device=bboxes.device)
            direction = 1

        a = (leye - point_3rd).square().sum().sqrt().item()
        b = (reye - point_3rd).square().sum().sqrt().item()
        c = (reye - leye).square().sum().sqrt().item()

        if b != 0 and c != 0:
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            cos_a = min(1.0, max(-1.0, cos_a))

            angle = math.acos(cos_a)
            angle = (angle * 180) / math.pi
            if direction == -1:
                angle = 90 - angle

            crop = crop.rotate(direction * angle)

        # update crops
        crops.append(crop)
    return crops
