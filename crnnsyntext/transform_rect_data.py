import os
from glob import glob
import cv2
import json
import numpy as np

# import matplotlib.pyplot as plt
from PIL import Image

PRE_DATA_PATH = "../ocr_data/ReCTS"
OUTPUT_PATH = "../ocr_data/ocr_syn_text"

PRE_IMAGE_PATH = 'img'
PRE_GT_PATH = 'gt'


def crop_rect(img, rect, alph=0.15):
    img = np.asarray(img)
    # get the parameter of the small rectangle
    # print("rect!")
    # print(rect)
    center, size, angle = rect[0], rect[1], rect[2]
    min_size = min(size)
    print('=========================')
    print(center)
    print(size)
    print(angle)
    print('=========================')

    if (angle > -45):
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # angle-=270
        size = (int(size[0] + min_size * alph), int(size[1] + min_size * alph))
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # size = tuple([int(rect[1][1]), int(rect[1][0])])
        img_rot = cv2.warpAffine(img, M, (width, height))
        # cv2.imwrite("debug_im/img_rot.jpg", img_rot)
        img_crop = cv2.getRectSubPix(img_rot, size, center)
        # print(img_rot)
        img_crop = Image.fromarray(img_crop)
    else:
        center = tuple(map(int, center))
        size = tuple([int(rect[1][1]), int(rect[1][0])])
        size = (int(size[0] + min_size * alph), int(size[1] + min_size * alph))
        angle -= 270
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        # cv2.imwrite("debug_im/img_rot.jpg", img_rot)
        img_crop = cv2.getRectSubPix(img_rot, size, center)
        img_crop = Image.fromarray(img_crop)
    return img_crop


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
        tags.append(content_dict['transcription'])
    f.close()
    return np.array(bboxes), tags

def split_crop_img(image_name):
    basename = os.path.basename(image_name)[:-4]
    img = get_img(image_name)

    # plt.imshow(img)
    # plt.show()
    gt_name = os.path.join(PRE_DATA_PATH, PRE_GT_PATH, basename + ".json")
    print(gt_name)

    bboxes, tags = get_bboxes(img, gt_name)

    gt_text = np.zeros(img.shape[0:2], dtype='uint8')
    bbox_list = []
    rects = []
    bboxes_image_text = []
    if bboxes.shape[0] > 0:
        # 调整保证bbox的坐标与缩放后的图片大小一致
        bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 4),
                            (bboxes.shape[0], int(bboxes.shape[1] / 2), 2)).astype('int32')
        for i in range(bboxes.shape[0]):
            cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
            # print('<============bbox=============>')
            # print(bboxes[i])
            rect = cv2.minAreaRect(bboxes[i])
            # print('<============rect=============>')
            # print(rect)
            # print('<============bPbox============>')
            bbox = cv2.boxPoints(rect)
            # print(bbox)

            bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
            rects.append(rect)

            # degree, w, h, cx, cy = rect
            (cx, cy), (h, w), degree = rect

            if cx < 0 or cy < 0 or h <= 0 or w <=0:
                print('params error, next image')
                continue

            # partImg, newW, newH = rotate_cut_img(im,  90  + degree  , cx, cy, w, h, leftAdjust, rightAdjust, alph)
            print('=============================')
            print(tags[i])
            partImg = crop_rect(img, ((cx, cy), (h, w), degree))
            # plt.imshow(partImg)
            # plt.show()

            newW, newH = partImg.size
            partImg_array = np.uint8(partImg)
            if newH > 1.5 * newW:
                partImg_array = np.rot90(partImg_array, 1)
            partImg = Image.fromarray(partImg_array).convert("RGB")
            partImg = partImg.convert('L')

            if not os.path.exists(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)

            partImg_name = os.path.join(OUTPUT_PATH, basename + ('_%d_.jpg' % i))
            partImg.save(partImg_name)
            bboxes_image_text.append((partImg_name, tags[i]))
    with open(os.path.join(OUTPUT_PATH, 'image_text_index.txt'), 'a+') as f:
        for (image_name, text) in bboxes_image_text:
            f.write(image_name + " " + text + "\n")
        f.close()


if __name__ == '__main__':
    index = 0
    for image_name in glob(os.path.join(PRE_DATA_PATH, PRE_IMAGE_PATH, "*.jpg")):
        print(image_name + " " + str(index))
        split_crop_img(image_name)
    # split_crop_img('../ocr_data/ReCTS/img/train_ReCTS_000275.jpg')





