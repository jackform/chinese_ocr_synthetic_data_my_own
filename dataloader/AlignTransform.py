import cv2
import numpy as np
import random
import torchvision.transforms as transforms
import torch

"""
保证缩放后与指定宽高不一致的图片中，随机切出和指定宽高一致的图片
"""


def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs

    # TODO 这里应当每次都做随机crop处理
    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # 取出图中连通区域的坐标，并取出最小(处于最左上的连通区域的左上角)的坐标，减去指定宽高
        tl = np.min(np.where(imgs[1] > 0), axis=1) - img_size
        # 保证不超出图片左上
        tl[tl < 0] = 0
        # 取出图中连通区域的坐标，并取出最大(处于最右下的连通区域的右下角)的坐标，减去指定宽高
        br = np.max(np.where(imgs[1] > 0), axis=1) - img_size
        # 保证不超出图片左上
        br[br < 0] = 0
        # 保证不超出图片右下
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)

        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs


def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


class AlignTransform(object):
    def __init__(self, img_size=224, min_scale=0.4):
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.min_scale = min_scale
        self.need_transform = True

    def __call__(self, batch):
        img, gt_text, gt_kernals, training_mask = zip(*batch)
        # if self.need_transform:
        if self.need_transform:
            imgs = [img, gt_text, training_mask]
            imgs.extend(gt_kernals)

            # TODO horizontal_flip是不是不需要
            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop(imgs, self.img_size)

            img, gt_text, training_mask, gt_kernals = imgs[0], imgs[1], imgs[2], imgs[3:]

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).float()
        gt_kernals = torch.from_numpy(gt_kernals).float()
        training_mask = torch.from_numpy(training_mask).float()

        return img, gt_text, gt_kernals, training_mask
