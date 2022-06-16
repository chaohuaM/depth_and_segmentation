# @Author  : ch
# @Time    : 2022/6/13 下午4:20
# @File    : dataaugment.py

import os
import numpy as np
import albumentations as A
import cv2


def save_png(img_path, data):
    cv2.imwrite(img_path, data)


transform = A.Compose([
    A.RandomSizedCrop(min_max_height=(300, 500), height=512, width=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
        A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
        A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
    ], p=0.9),
    # A.ShiftScaleRotate(border_mode=0),
    # 随机应用仿射变换：平移，缩放和旋转输入
    A.RandomBrightnessContrast(p=0.5),  # 随机明亮对比度
    # A.ColorJitter(p=0.2),
    A.RandomGamma(p=0.5),
])

dataset_path = '../dataset/MSL'
save_dir = os.path.join(dataset_path, 'aug_data')

image_dirname = 'rgb'
img_dir = os.path.join(dataset_path, image_dirname)
img_suffix = "." + os.listdir(img_dir)[0].split('.')[-1]

mask_dirnames = ['semantic_01_label', 'rock_label_vis', 'inv-depth-png', 'inv-depth-npy']
aug_nums = 2

txt_path = os.path.join(dataset_path, "ImageSets/train.txt")

with open(txt_path, 'r') as f:
    train_lines = f.readlines()

new_img_dir = os.path.join(save_dir, image_dirname)
if not os.path.exists(new_img_dir): os.makedirs(new_img_dir)

save_mask_dirnames = []
for dirname in mask_dirnames:
    new_mask_dir = os.path.join(save_dir, dirname)

    if not os.path.exists(new_mask_dir): os.makedirs(new_mask_dir)
    save_mask_dirnames.append(new_mask_dir)

for count in range(len(train_lines)):
    img_name = train_lines[count][:-1]

    img_path = os.path.join(img_dir, img_name + img_suffix)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = []
    for dirname in mask_dirnames:
        mask_dir = os.path.join(dataset_path, dirname)
        mask_suffix = "." + os.listdir(mask_dir)[0].split('.')[-1]
        mask_path = os.path.join(mask_dir, img_name+mask_suffix)

        if mask_suffix == ".npy":
            mask = np.load(mask_path)
        else:
            mask = cv2.imread(mask_path, -1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        masks.append(mask)

    for i in range(aug_nums):

        transformed = transform(image=image, masks=masks)

        transformed_image = transformed['image']
        transformed_masks = transformed['masks']

        save_name = img_name + '_' + str(i) + '.png'

        save_png(os.path.join(new_img_dir, save_name), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

        for i, save_mask_dirname in enumerate(save_mask_dirnames):
            if 'npy' in save_mask_dirname:
                np.save(os.path.join(save_mask_dirname, save_name.replace('.png', '.npy')), transformed_masks[i])
            else:
                save_png(os.path.join(save_mask_dirnames[i], save_name), cv2.cvtColor(transformed_masks[i], cv2.COLOR_RGB2BGR))

    if count % 10 == 0:
        print(count)
