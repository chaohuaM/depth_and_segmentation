import os

import cv2
import numpy as np
from torch.utils.data.dataset import Dataset

from utils.utils import preprocess_input, load_exr, resize_image, paste_image


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
        img_item = cv2.resize(img_item, (nw, nh))

        if flip_flag:
            img_item = cv2.flip(img_item, flipCode)

        # place img_item
        transform_item = paste_image(img_item, new_img_item, dx, dy)
        transform_items.append(transform_item)

        return transform_items


def pixel_level_distort(image, hue, sat, val):
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
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
        原图路径     数据集文件夹/rgb/*_rgb_*.png
        标签路径     数据集文件夹/semantic_01_label/*_semantic_label_*.png
        深度图路径    数据集文件夹/depth/*_pinhole_depth_*.exr
        '''
        img_path = os.path.join(os.path.join(self.dataset_path, "rgb"), name + ".png")
        label_path = os.path.join(os.path.join(self.dataset_path, "semantic_01_label"),
                                  name.replace('rgb_00', 'semantic_label_01') + ".png")
        depth_img_path = os.path.join(os.path.join(self.dataset_path, "depth_exr"),
                                      name.replace('rgb', 'pinhole_depth') + ".exr")

        x_img = cv2.imread(img_path, -1)
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        y_label = cv2.imread(label_path, -1)
        depth_img = load_exr(depth_img_path)
        # -------------------------------#
        #   数据增强
        # -------------------------------#
        x_img, y_label, depth_img = self._get_random_data(x_img, y_label, depth_img)

        x_img = np.transpose(preprocess_input(np.array(x_img, np.float64)), [2, 0, 1])
        y_label = np.array(y_label)
        y_label[y_label >= self.num_classes] = self.num_classes
        # -------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        # -------------------------------------------------------#
        seg_labels = np.eye(self.num_classes + 1)[y_label.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        depth_img = np.array(depth_img, np.float64)

        return x_img, y_label, seg_labels, depth_img

    def _get_random_data(self, image, label, depth, jitter=.3, hue=.1, sat=1.5, val=1.5):
        h, w = self.input_shape

        if not self.transform:
            new_image = resize_image(image, [h, w])
            new_label = resize_image(label, [h, w])
            new_depth = resize_image(depth, [h, w])
            return new_image, new_label, new_depth

        # 是否翻转
        flip = rand() < .5
        # resize image

        img_items = [image, label, depth]

        # image-level, 整体变换对所有图像都要操作
        image, label, depth = image_level_transform(img_items, flip, jitter, self.input_shape)

        # pixel-level distortion image
        image = pixel_level_distort(image, hue, sat, val)

        return image, label, depth


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


