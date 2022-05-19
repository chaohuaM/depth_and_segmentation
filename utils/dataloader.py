import os

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils.utils import preprocess_input, load_exr, resize_and_centered, paste_image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def image_level_transform(img_items, flip_flag, jitter, size):
    h, w = size
    rand_jit1 = rand(1 - jitter, 1 + jitter)
    rand_jit2 = rand(1 - jitter, 1 + jitter)
    new_ar = w / h * rand_jit1 / rand_jit2

    scale = rand(0.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)

    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))

    transform_items = []
    flipCode = np.random.choice([1, 0, -1], 1)[0]

    for img_item in img_items:

        new_img_item = np.zeros_like(img_item)
        new_img_item = cv2.resize(new_img_item, (w, h))
        img_item = cv2.resize(img_item, (nw, nh))

        if flip_flag:
            img_item = cv2.flip(img_item, flipCode)

        # place img_item
        transform_item = paste_image(img_item, new_img_item, dx, dy)
        transform_items.append(transform_item)

    return transform_items


def pixel_level_distort(image, hue=1, sat=2, val=2):
    hue = rand(-hue, hue) if rand() < .8 else 0
    sat = rand(0, sat) if rand() < .8 else 1
    val = rand(0, val) if rand() < .8 else 1

    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    # x[..., 1] = np.power(x[..., 1], sat)
    # x[..., 2] = np.power(x[..., 2], val)
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    # image_data = x.astype(np.uint8)
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

    return image_data


class RockDataset(Dataset):
    def __init__(self, img_path_lines, input_shape, num_classes, transform, dataset_path):
        super(RockDataset, self).__init__()
        self.path_lines = img_path_lines
        self.length = len(img_path_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.transform = transform
        self.dataset_path = dataset_path
        # self.image_level_transform = True
        self.color_jitter = True
        self.img_dir = os.path.join(self.dataset_path, 'rgb')
        self.label_dir = os.path.join(self.dataset_path, 'semantic_01_label')
        self.depth_dir = os.path.join(self.dataset_path, 'inv-depth-01-npy')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        返回一个单位的数据
        :param index:  数据的索引
        :return: 包括原图，mask，one-hot格式标签，深度图
        """
        path_line = self.path_lines[index]
        name = path_line.split()[0]

        # -------------------------------#
        #   从文件中读取图像
        # -------------------------------#
        '''
        原图路径     数据集文件夹/*.png
        标签路径     数据集文件夹/semantic_01_label/*.png
        深度图路径    数据集文件夹/depth/*_pinhole_depth_*.exr
        '''
        img_path = os.path.join(self.img_dir, name + ".png")
        label_path = os.path.join(self.label_dir, name + ".png")
        # 最好使用逆深度，而且进行归一化『0，1』
        depth_img_path = os.path.join(self.depth_dir, name + ".npy")

        # 以50%概率读取成灰度图 再转换成rgb，只有亮度信息
        if self.input_shape[2] == 1 or (self.transform and rand() < 0.5):
            x_img = cv2.imread(img_path, 0)
            x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2RGB)
        else:
            x_img = cv2.imread(img_path, 1)
            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)

        y_label = cv2.imread(label_path, 0)
        depth_img = np.load(depth_img_path)
        # -------------------------------#
        #   数据增强
        # -------------------------------#
        x_img, y_label, depth_img = self._get_random_data(x_img, y_label, depth_img)

        if self.input_shape[2] == 1:
            x_img = cv2.cvtColor(x_img, cv2.COLOR_RGB2GRAY)
            x_img = x_img[:, :, np.newaxis]

        x_img = np.transpose(preprocess_input(np.array(x_img, np.float32)), [2, 0, 1])

        y_label = np.array(y_label)
        if len(np.unique(y_label))-1 > self.num_classes:
            raise ValueError(
                "num_classes is {}, but the {} has {} classes.".format(
                    self.num_classes, label_path, len(np.unique(y_label)))
            )
        y_label = y_label[np.newaxis, :, :]

        # 读取深度信息
        depth_img = np.array(depth_img, np.float32)
        depth_img = depth_img[np.newaxis, :, :]
        # -------------------------------------------------------#
        #   转化成one_hot的形式
        # -------------------------------------------------------#
        # seg_labels = np.eye(self.num_classes+1)[y_label.reshape([-1])][:, 1:]  # 丢弃第一类，是背景类
        # seg_labels = seg_labels.reshape((self.num_classes, int(self.input_shape[0]), int(self.input_shape[1])))

        # return x_img, y_label, seg_labels, depth_img
        return x_img, y_label, depth_img

    def _get_random_data(self, image, label, depth, jitter=.3, hue=.1, sat=1.5, val=1.5):
        h, w, c = self.input_shape[0], self.input_shape[1], self.input_shape[2]

        if not self.transform:
            new_image = resize_and_centered(image, [h, w])
            new_label = resize_and_centered(label, [h, w])
            new_depth = resize_and_centered(depth, [h, w])
            return new_image, new_label, new_depth

        # 是否翻转
        flip = rand() < .5
        # resize image

        img_items = [image, label, depth]

        # image-level, 整体变换对所有图像都要操作
        image, label, depth = image_level_transform(img_items, flip, jitter, [h, w])

        # pixel-level distortion image
        if self.color_jitter:
            image = pixel_level_distort(image, hue, sat, val)

        return image, label, depth


class RealRockDataset(RockDataset):
    def __init__(self, img_path_lines, input_shape, num_classes, transform, dataset_path):
        super().__init__(img_path_lines, input_shape, num_classes, transform, dataset_path)

        # 关闭颜色抖动的增强部分
        self.color_jitter = False
        self.img_dir = os.path.join(self.dataset_path, 'images')
        self.label_dir = os.path.join(self.dataset_path, 'label_mask')
        self.depth_dir = os.path.join(self.dataset_path, 'inv-depth-npy')


# DataLoader中collate_fn使用
def rock_dataset_collate(batch):
    """
    调用时按batch_size返回数据集
    :param batch: int， batch_size
    :return: 返回格式为__get_item__返回的数据内容的复数
    """
    images = []
    masks = []
    seg_labels = []
    depths = []
    for img, mask, label, depth in batch:
        images.append(img)
        masks.append(mask)
        seg_labels.append(label)
        depths.append(depth)
    images = np.array(images)
    masks = np.array(masks)
    seg_labels = np.array(seg_labels)
    depths = np.array(depths)
    return images, masks, seg_labels, depths


def rock_dataset_collate_pl(batch):
    """
    调用时按batch_size返回数据集, 在使用pytorch-ligntning调用
    :param batch: int， batch_size
    :return: 返回格式为__get_item__返回的数据内容的复数
    """
    images = []
    masks = []
    seg_labels = []
    depths = []
    for img, mask, label, depth in batch:
        images.append(img)
        masks.append(mask)
        seg_labels.append(label)
        depths.append(depth)
    images = torch.tensor(images)
    masks = torch.tensor(masks)
    masks = torch.unsqueeze(masks, 1)
    seg_labels = torch.tensor(seg_labels)
    seg_labels = torch.movedim(seg_labels, -1, 1)
    depths = torch.tensor(depths)
    return images, masks, seg_labels, depths
