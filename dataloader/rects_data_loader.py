import json
import glob
import os
import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg

import matplotlib.pyplot as plt

# ctw_root_dir = '../OCR/dataset/'
ctw_root_dir = 'data/'
ctw_train_data_dir = ctw_root_dir + 'ReCTS/img/'
ctw_train_gt_dir = ctw_root_dir + 'ReCTS/gt/'
# ctw_train_gt_dir = ctw_root_dir + 'ReCTS/gt_unicode/'
ctw_test_data_dir = ctw_root_dir + 'test/text_image/'
ctw_test_gt_dir = ctw_root_dir + 'test/text_label_curve/'
random.seed(123456)


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


def random_scale(img, min_size):
    h, w = img.shape[0:2]
    # 确保不大于1280
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img.shape[0:2]
    # TODO 考虑去除随机缩放？
    random_scale = np.array([0.5, 1.0, 2.0, 3.0])
    scale = np.random.choice(random_scale)
    # 确保不小于训练指定的宽高
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


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


def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bboxes.append(shrinked_bbox)

    return np.array(shrinked_bboxes)


class ReCTSDataLoader(data.Dataset):
    def __init__(self, need_transform=False, img_size=224, kernel_num=7, min_scale=0.4, train_data_dir=None, train_gt_dir=None):
        self.need_transform = need_transform

        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.kernel_num = kernel_num
        self.min_scale = min_scale

        if train_data_dir is None and train_gt_dir is None:
            self.ctw_train_data_dir = ctw_train_data_dir
            self.ctw_train_gt_dir = ctw_train_gt_dir
        else:
            self.ctw_train_data_dir = train_data_dir
            self.ctw_train_gt_dir = train_gt_dir

        data_dirs = [self.ctw_train_data_dir]
        gt_dirs = [self.ctw_train_gt_dir]

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = glob.glob(os.path.join(data_dir, '*.jpg'))

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_name = os.path.basename(img_name)
                img_path = data_dir + img_name
                img_paths.append(img_path)

                gt_name = img_name.split('.')[0] + '.json'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

    @staticmethod
    def get_bboxes(img, gt_path):
        h, w = img.shape[0:2]
        # lines = util.io.read_lines(gt_path)
        f = open(gt_path, 'r')
        line = f.readline()
        bboxes = []
        tags = []
        # assert len(lines) != 1
        # line = lines[0]
        contents_dict = json.loads(line)
        for content_dict in contents_dict['lines']:
            bbox = np.asarray(content_dict['points'])
            # 先除以w, h。 后面对应的img会随机缩放，缩放后会再乘以缩放后的宽高
            bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 4)
            bboxes.append(bbox)
            tags.append(True)
        return np.array(bboxes), tags

    @staticmethod
    def imshow(img):
        # img = img / 2 + 0.5
        # npimg = img
        # plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.imshow(img)
        plt.show()

    @staticmethod
    def get_img(img_path):
        try:
            img = cv2.imread(img_path)
            # 读取出来的是GBR, 需要转换成RGB
            img = img[:, :, [2, 1, 0]]
        except Exception as e:
            print(str(e))
            print(img_path)
            raise
        return img

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = self.get_img(img_path)
        bboxes, tags = self.get_bboxes(img, gt_path)

        if self.need_transform:
            img = random_scale(img, self.img_size[0])

        # self.imshow(img)

        gt_text = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            # 调整保证bbox的坐标与缩放后的图片大小一致
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 4),
                                (bboxes.shape[0], int(bboxes.shape[1] / 2), 2)).astype('int32')
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
                if not tags[i]:
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)
        # self.imshow(gt_text)

        gt_kernals = []
        for i in range(1, self.kernel_num):
            # TODO min_scale考虑改成0.31
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            # exam: {min_scale:0.4, kernel_num:7} rate:[ 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4 ]
            gt_kernal = np.zeros(img.shape[0:2], dtype='uint8')
            kernal_bboxes = shrink(bboxes, rate)
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_kernal, [kernal_bboxes[i]], -1, 1, -1)
            gt_kernals.append(gt_kernal)

        if self.need_transform:
            imgs = [img, gt_text, training_mask]
            imgs.extend(gt_kernals)

            # TODO horizontal_flip是不是不需要
            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop(imgs, self.img_size)

            img, gt_text, training_mask, gt_kernals = imgs[0], imgs[1], imgs[2], imgs[3:]
            # img, gt_text, training_mask, gt_kernals = imgs[0], imgs[1], imgs[2], imgs[3]

        # 上面是有区分连通区域，这里统一设置为1
        gt_text[gt_text > 0] = 1
        gt_kernals = np.array(gt_kernals)

        if self.need_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).float()
        gt_kernals = torch.from_numpy(gt_kernals).float()
        training_mask = torch.from_numpy(training_mask).float()

        return img, gt_text, gt_kernals, training_mask
