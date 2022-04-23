# coding: utf-8
# @Author：@mls， @ch

import colorsys
from PIL import Image
import cv2
import numpy as np


def change_color(image_path, save_path, mask_path):

    # 目标色调值
    target_hue = 16/360
    target_s = 0.50
    target_v = 0.59

    # 读入图片，转化为 RGB 色值
    image = Image.open(image_path).convert('RGB')
    # iamge_h = Image.open(filename1).convert('RGB')
    # 将 RGB 色值分离
    mask = Image.open(mask_path)
    r, g, b = image.split()
    result_r, result_g, result_b = [], [], []

    # image1.load()
    # r1, g1, b1 = image.split()
    # result_r1, result_g1, result_b1 = [], [], []

    # 依次对每个像素点进行处理
    for pixel_r, pixel_g, pixel_b, mask_value in zip(r.getdata(), g.getdata(), b.getdata(), mask.getdata()):

        if mask_value == 0:
            target_hue = 16/360
        else:
            target_hue = 0.0952
            # 转为 HSV 色值
        h, s, v = colorsys.rgb_to_hsv(pixel_r / 255., pixel_b / 255., pixel_g / 255.)
        # 转回 RGB 色系
        rgb = colorsys.hsv_to_rgb(target_hue, s*1.5, v)
        pixel_r, pixel_g, pixel_b = [int(x * 255.) for x in rgb]

        # 每个像素点结果保存
        result_r.append(pixel_r)
        result_g.append(pixel_g)
        result_b.append(pixel_b)

    r.putdata(result_r)
    g.putdata(result_g)
    b.putdata(result_b)

    # 合并图片
    image = Image.merge('RGB', (r, g, b))
    # 输出图片
    image.save(save_path)


# 使用opencv库操作速度比Image库速度快很多，特别是大量数据的时候
def change_color_opencv(image_path, save_path, mask_path):
    """
    利用opencv进行读取并进行颜色的转换，其中mask为天空的mask，sky为1，non-sky为0，
    :param image_path: str，输入图片路径
    :param save_path: str，输入保存路径
    :param mask_path: str，输入天空掩膜图像
    :return:
    """
    # 读取并转换到hsv空间
    image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_BGR2HSV)
    # 分离通道
    h, s, v = cv2.split(image)

    # 读取sky mask，～为取反操作，所以要先转为类型bool
    sky = cv2.imread(mask_path, 0)
    non_sky = ~sky.astype('bool')

    # 分别对h， s， v 进行赋值和变换, 前面的数值是观察真实图片得到的
    new_h = 0.0912 * 360 * sky + 0.0489 * 360 * non_sky
    # new_s = 0.900 * s * sky + 1.5 * s * non_sky
    new_s = 0.3714 * sky + 0.5455 * non_sky
    new_v = 0.9 * v * sky + 0.975 * v * non_sky

    # h <=360, s，v <=1
    new_h[new_h > 360] = 360
    new_s[new_s > 1] = 1
    new_v[new_v > 1] = 1

    # 因为上一步操作完成后数据类型不一样，进行转换
    new_h = new_h.astype(h.dtype)
    new_s = new_s.astype(h.dtype)
    new_v = new_v.astype(h.dtype)

    # 合并通道并转换为rgb空间，最后要乘回255
    output = cv2.merge([new_h, new_s, new_v])
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)*255

    # 保存图像
    cv2.imwrite(save_path, output)



