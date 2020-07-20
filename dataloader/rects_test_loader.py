import os

from PIL import Image
from torch.utils import data
import cv2
import random
from glob import glob
import torchvision.transforms as transforms

ctw_root_dir = './data/'
ctw_test_data_dir = ctw_root_dir + 'ReCTS/test/'

random.seed(123456)


def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print('read image error:'+img_path)
        # print(img_path)
        raise e
    return img


def scale(img, long_size=1280):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


class ReCTSTestDataloader(data.Dataset):
    def __init__(self, long_size=1280):

        data_dirs = [ctw_test_data_dir]

        self.img_paths = []

        for data_dir in data_dirs:
            # img_names = util.io.ls(data_dir, '.jpg')
            img_names = glob(os.path.join(data_dir, '*.jpg'))
            # img_names.extend(util.io.ls(data_dir, '.png'))
            # img_names.extend(util.io.ls(data_dir, '.gif'))

            img_paths = []
            for idx, img_name in enumerate(img_names):
                # img_name =
                # img_path = data_dir + img_name
                img_path = img_name
                img_paths.append(img_path)

            self.img_paths.extend(img_paths)

        # self.img_paths = self.img_paths[440:]
        # self.gt_paths = self.gt_paths[4540:]
        self.long_size = long_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path)

        scaled_img = scale(img, self.long_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)

        return img[:, :, [2, 1, 0]], scaled_img
