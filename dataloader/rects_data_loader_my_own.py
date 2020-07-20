import json

from torch.utils import data
import os
import cv2
import numpy as np
from glob import glob
import pyclipper
import Polygon as plg


class ReCTSDataLoader(data.Dataset):

    def __init__(self, img_dir, gt_dir, kernel_num=7, img_size=640, min_scale=0.31, is_training=True):
        self.kernel_num = kernel_num
        self.img_size = 640
        self.min_scale = 0.31

        self.gt_paths = []
        self.image_paths = glob(os.path.join(img_dir, '*.jpg'))
        self.data_num = len(self.image_paths)
        for idx, img_name in enumerate(self.image_paths):
            # 同步保存gt信息
            img_name = os.path.basename(img_name)
            gt_name = img_name.split('.')[0] + '.json'
            gt_path = gt_dir + gt_name
            self.gt_paths.append(gt_path)

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        image_name = self.image_paths[index]
        img = self.get_image(image_name)

        gt_name = self.gt_paths[index]

        bboxes, tags = self.get_bboxs(img, gt_name)

        img = random_scale(img, self.img_size[0])

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

        gt_text[gt_text > 0] = 1
        gt_kernals = np.array(gt_kernals)

        return img, gt_text, gt_kernals, training_mask

    @staticmethod
    def get_image(image_name):
        try:
            img = cv2.imread(image_name)
            img = img[:, :, [2, 1, 0]]
        except Exception as e:
            print(str(e))
            print(image_name)
            img = None
        return img

    @staticmethod
    def get_bboxs(gt_path, img):
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


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
