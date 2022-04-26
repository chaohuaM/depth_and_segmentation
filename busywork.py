import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.optim as optim
import glob
from predict_model import blend_image, PredictModel


def normalization(data):
    mi = np.min(data)
    ma = np.max(data)
    _range = ma
    return data / _range


def save_png(img_path, data):
    cv2.imwrite(img_path, data)


def exr2png(exr_path, png_path):
    # 使用openexr包读取
    # tiff_file = exr2tiff(file_path)
    # 使用opencv 读取
    image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image[image == np.max(image)] = 0
    image[image > 10] = 10

    # img1 = image
    img2 = normalization(image) * 255
    img = cv2.applyColorMap(img2.astype(np.uint8), cv2.COLORMAP_MAGMA)

    # img3 = Z_ScoreNormalization(image)
    #
    # eq1 = cv2.equalizeHist(img1.astype('uint8'))
    # eq2 = cv2.equalizeHist((img2 * 255).astype('uint8'))
    # eq3 = cv2.equalizeHist((img3 * 255).astype('uint8'))

    # cv2.imshow('depth', img)
    # cv2.waitKey()

    save_png(png_path, img)


# # 转换深度图
# file_dir = '/home/ch5225/Desktop/模拟数据/2022-02-02-00-23-59'
#
# depth_exr_dir = file_dir + '/depth_exr'
# depth_png_dir = file_dir + '/depth_10m'
#
# if not os.path.exists(depth_png_dir): os.makedirs(depth_png_dir)
#
# exr_list = glob.glob(depth_exr_dir + '/*')
# count = 0
# for exr_img in exr_list:
#     img_name = exr_img.split('/')[-1]
#     png_path = depth_png_dir + '/' + img_name.replace('exr', 'png')
#
#     exr2png(exr_img, png_path)
#
#     count += 1
#     if count % 100 == 0: print(count)


# 将标签中的rgb值转为为0和1
# import glob
#
# label_dir = '/home/ch5225/Desktop/模拟数据/oaisys-new/semantic_00/'
# new_label_dir = '/home/ch5225/Desktop/模拟数据/oaisys-new/sky=1/'
#
# if not os.path.exists(new_label_dir): os.makedirs(new_label_dir)
#
# count = 0
#
# for img_path in glob.glob(label_dir + '*.png'):
#     img_name = img_path.split('/')[-1]
#     img = cv2.imread(img_path, 0)
#     h, w = img.shape
#
#     label = np.zeros([h, w], dtype=np.uint8)
#     label[img >= 45] = 1
#
#     save_png(new_label_dir + '/' + img_name, label)
#
#     count += 1
#     if count % 100 == 0:
#         print(count)


# # 将标签中的rgb值转为为0和1
# img_dir = '/home/ch5225/Desktop/模拟数据/oaisys-new/rgb/'
# label_dir = '/home/ch5225/Desktop/模拟数据/oaisys-new/semantic_01/'
# new_label_dir = '/home/ch5225/Desktop/模拟数据/oaisys-new/semantic_01_label/'
# label_vis_dir = '/home/ch5225/Desktop/模拟数据/oaisys-new/semantic_01_label_vis/'
#
# if not os.path.exists(new_label_dir): os.makedirs(new_label_dir)
# if not os.path.exists(label_vis_dir): os.makedirs(label_vis_dir)
#
# count = 0
# for img_path in glob.glob(label_dir + '*.png'):
#     img_name = img_path.split('/')[-1]
#     img = cv2.imread(img_path, 0)
#     h, w = img.shape
#
#     label = np.zeros([h, w], dtype=np.uint8)
#     label[img >= 100] = 1
#
#     save_png(new_label_dir + '/' + img_name, label)
#
#     save_png(new_label_dir + '/' + img_name, label)
#
#     raw_img = cv2.imread(img_dir + '/' + img_name)
#     vis_img = blend_image(raw_img, label)
#
#     save_png(label_vis_dir + '/' + img_name, vis_img)
#
#     count += 1
#
#     if count % 200 == 0:
#         print(count)


# 测试dataloader函数
# from torch.utils.data import DataLoader
# from utils.dataloader import RockDataset, rock_dataset_collate, rock_dataset_collate_pl
#
# train_lines = ['1306sensorRight', '4111sensorRight', '0752sensorLeft', '1080sensorLeft',
#                '3940sensorLeft', '4465sensorRight', '1665sensorRight', '3687sensorLeft',
#                '2669sensorLeft', '1702sensorLeft', '4359sensorLeft', '4550sensorRight']
# input_shape = [256, 256]
# input_channel = 1
# input_shape.append(input_channel)
# num_classes = 1
# data_dir = '/home/ch5225/Desktop/模拟数据/2022-02-02-00-23-59'
# train_dataset = RockDataset(train_lines, input_shape, num_classes, True, data_dir)
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=4, pin_memory=True,
#                               drop_last=True)
#
# for gen in train_dataloader:
#     print(gen[0].size())

# 测试pytorch-lightning 方式训练
import pytorch_lightning as pl
from model.create_model_pl import MyModel

# model = MyModel('Unet', encoder_name='resnet50', in_channels=3, num_classes=2, encoder_weights='imagenet')
# trainer = pl.Trainer(gpus=1, max_epochs=100)
#
# trainer.fit(
#     model,
#     train_dataloaders=train_dataloader,
# )

# 测试pytorch-lightning test
# moel = MyModel.load_from_checkpoint(checkpoint_path='logs/Unet/2022_04_07_11_13_21/checkpoints/epoch=34-step=455.ckpt',
#                                     hparams_file='logs/Unet/2022_04_07_11_13_21/hparams.yaml'
#                                     ).model

#
# for batch in gen:
#     depths = torch.from_numpy(batch[3]).type(torch.FloatTensor)
#     depths = torch.unsqueeze(depths, 1)

# 测试gradient difference loss
# from losses.grad_loss import GRAD_LOSS, imgrad_yx
#
# loss1 = imgrad_yx(depths)
# loss = GRAD_LOSS(depths, depths).item()

# 测试模型
# from model.unet_with_backbone import Unet
#
#
# model = Unet(backbone='resnet50', deformable_mode=False).cuda()
# x = torch.zeros(2, 3, 512, 512).cuda()
# y, y_depth = model(x)
# #
# model_train = model.train()
# # 测试是否可以训练
# from utils.callbacks import LossHistory
# from utils.utils_fit import fit_one_epoch_no_val
#
# batch_size = 2
# cls_weights = np.ones([num_classes], np.float32)
# loss_history = LossHistory("logs/")
# epoch_step = len(train_lines) // batch_size
#
# optimizer = optim.Adam(model_train.parameters(), 1e-4)
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
#
# for epoch in range(0, 50):
#     fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, 50, True, False,
#                          False, cls_weights, num_classes)
#     lr_scheduler.step()

#
# from utils.callbacks import LossHistory
# import datetime
#
# time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
# loss_history = LossHistory("logs/", time_str)
#
# print(loss_history.time_str)

# # 查看深度图
# from utils.utils import load_exr
#
# left_exr_path = '/home/ch5225/Desktop/模拟数据/2022-02-02-00-23-59/depth_exr/0100sensorLeft_pinhole_depth_00.exr'
# right_exr_path = '/home/ch5225/Desktop/模拟数据/2022-02-02-00-23-59/depth_exr/0100sensorRight_pinhole_depth_00.exr'
#
# left_depth = load_exr(left_exr_path)
# right_depth = load_exr(right_exr_path)
#
# img_path = '/home/ch5225/Desktop/5.NaTeCam-2C/第五批/HX1-Ro_GRAS_NaTeCamB-F-002_SCI_N_20211029062325_20211029062325_00164_A.2C.jpg'
# #
# img_bgr = cv2.imread(img_path, 0)
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
# # img_g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# # img_g = img_g[:, :, np.newaxis]
#
# input_shape = [512, 512]
# input_channel = 3

# input_shape.append(input_channel)
#
# # img_rgb = cv2.cvtColor(img_g, cv2.COLOR_GRAY2BGR)
#
# from utils.dataloader import pixel_level_distort
#
# img_t = pixel_level_distort(img_rgb).astype(np.uint8)
# img_t1 = cv2.cvtColor(img_t, cv2.COLOR_RGB2BGR).astype(np.uint8)
#
# plt.subplot(221)
# plt.imshow(img_rgb)
# plt.axis('off')
# plt.subplot(222)
# plt.imshow(img_bgr)
# plt.axis('off')
# plt.subplot(223)
# plt.imshow(img_t)
# plt.axis('off')
# plt.subplot(224)
# plt.imshow(img_t1)
# plt.axis('off')
# plt.show()


# np.seterr(divide='ignore', invalid='ignore')
# left_disparity = 595.60 * 270 / (left_depth * 1000)
# right_disparity = 595.90 * 270 / (right_depth * 1000)


from argparse import ArgumentParser
import json

# parser = ArgumentParser()
# parser.add_argument('--seed', type=int, default=8)
# parser.add_argument('--resume', type=str, default='a/b/c.ckpt')
# parser.add_argument('--surgery', type=str, default='190', choices=['190', '417'])
# args = parser.parse_args()
#
# with open('commandline_args.json', 'w') as f:
#     json.dump(args.__dict__, f, indent=2)

# parser = ArgumentParser()
# args = parser.parse_args()
# with open('basic_args.json', 'r') as f:
#     args.__dict__ = json.load(f)

'''
# 采用现成方法获得单目深度估计图
import cv2
import torch
from predict_model import show_depth

import matplotlib.pyplot as plt

img_dir = '/home/ch5225/chaohua/MarsData/Data/rockA&B/images/'
depth_dir = '/home/ch5225/chaohua/MarsData/Data/rockA&B/inv-depth-png/'
depth_npy_dir = '/home/ch5225/chaohua/MarsData/Data/rockA&B/inv-depth-npy/'

if not os.path.exists(depth_dir): os.makedirs(depth_dir)
if not os.path.exists(depth_npy_dir): os.makedirs(depth_npy_dir)


model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

count = 0
for img_path in glob.glob(img_dir + '*.png'):
    img_name = img_path.split('/')[-1]
    # img_path = '/home/ch5225/chaohua/lunar_rocky_landscape/images/render_clean/render1164.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    np.save(depth_npy_dir+'/'+img_name.replace('.png', '.npy'), output)

    depth_img = show_depth(output)
    save_png(depth_dir+'/'+img_name, depth_img)

    count += 1

    if count % 20 == 0:
        print(count)
'''

# pc = point_cloud_generator(focal_length=2383.60, scalingfactor=1.0)
#
# pc.rgb = img_raw
# pc.depth = pr_depth
# pc.calculate()
# pc.write_ply('pc1.ply')
# pc.show_point_cloud()

# from predict import pr_Unet
#
# pr_unet = pr_Unet(config_path='logs/2022_03_11_17_49_48/2022_03_11_17_49_48_config.yaml',
#                   model_weights_path='logs/2022_03_11_17_49_48/ep100.pth')
#
# image_path = '/home/ch5225/chaohua/oaisys/oaisys_tmp/2022-03-03-15-15-02/batch_0002/sensorLeft/0009sensorLeft_rgb_00.png'
#
# img = cv2.imread(image_path)
#
# pr_image, pr_depth = pr_unet.detect_image(img)
#
# plt.subplot(121)
# plt.imshow(pr_image)
# plt.subplot(122)
# plt.imshow(pr_depth)
# plt.show()


# # 将标签中的rgb值转为为0和1
# import glob
#
# label_dir = '/home/ch5225/chaohua/lunar_rocky_landscape/images/clean/'
# new_label_dir = '/home/ch5225/chaohua/lunar_rocky_landscape/images/labels/'
#
# if not os.path.exists(new_label_dir): os.makedirs(new_label_dir)
#
# count = 0
#
# for img_path in glob.glob(label_dir + '*.png')[:5]:
#     img_name = img_path.split('/')[-1]
#     img = cv2.imread(img_path, 1)
#     h, w, c = img.shape
#
#     label = np.zeros([h, w], dtype=np.uint8)
#     label[img[:, :, 0] == 255] = 1
#     label[img[:, :, 1] == 255] = 1
#     # label[img[:,:,1] > 100] = 1
#
#     save_png(new_label_dir + '/' + img_name, img)
#
#     count += 1
#
#     if count % 10 == 0:
#         print(count)


# from model.unet_with_backbone import Unet
# from torchsummary import summary
#
# model = Unet(backbone='resnet50')
# summary(model.to('cuda'), (3, 512, 512))


# 测试边缘检测代码

# image_path = '/home/ch5225/Desktop/组会插图/LocCam_2015_11_26_12_57_21_133_1_json.png'
# result_path = 'test.png'
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度图像
# image = cv2.resize(image, (512, 512))
# image = cv2.GaussianBlur(image, (3, 3), 0)
# cv2.imshow("Original", image)
# cv2.waitKey(100)

# 拉普拉斯边缘检测
# lap = cv2.Laplacian(image, cv2.CV_64F)  # 拉普拉斯边缘检测
# lap = np.uint8(np.absolute(lap))  ##对lap去绝对值
# cv2.imshow("Laplacian", lap)
# cv2.waitKey()

# # Sobel边缘检测
# sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # x方向的梯度
# sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # y方向的梯度
#
# sobelX = np.uint8(np.absolute(sobelX))  # x方向梯度的绝对值
# sobelY = np.uint8(np.absolute(sobelY))  # y方向梯度的绝对值
#
#
# sobelCombined = cv2.bitwise_or(sobelX, sobelY)  #
# cv2.imshow("Sobel X", sobelX)
# cv2.waitKey(100)
# cv2.imshow("Sobel Y", sobelY)
# cv2.waitKey(100)
# cv2.imshow("Sobel Combined", sobelCombined)
# cv2.waitKey(0)
#
# plt.subplot(221)
# plt.imshow(image)
# plt.axis('off')
# plt.subplot(222)
# plt.imshow(sobelX)
# plt.axis('off')
# plt.subplot(223)
# plt.imshow(sobelY)
# plt.axis('off')
# plt.subplot(224)
# plt.imshow(sobelCombined)
# plt.axis('off')
# plt.show()

# 使用其他模型
# import segmentation_models_pytorch as smp
# from torchsummary import summary
#
# model = smp.DeepLabV3(encoder_name='resnet18', decoder_channels=64, in_channels=1)
# model = smp.Unet(encoder_name='resnet18').decoder
#
# x = torch.zeros(1, 3, 512, 512)
# y = model(x)
# for u in y:
#     print(u.shape)
# summary(model.to('cuda'), (1, 512, 512))


# pth转变为onnx模型
# from utils.utils import pth2onnx
# from predict import PredictModel
# from model.unet_dual_decoder import UnetDualDecoder

# config_path = 'logs/2022_03_27_21_48_23/2022_03_27_21_48_23_config.yaml'
# pth_path = 'logs/2022_03_27_21_48_23/ep100.pth'
# onnx_path = pth_path.replace('.pth', '.onnx')
#
# net = PredictModel(config_path=config_path, model_weights_path=pth_path).net
# pth2onnx(model=net, input_shape=[1, 3, 512, 512], onnx_path=onnx_path)

# model = UnetDualDecoder(backbone='resnet18').cuda()
# pth2onnx(model, input_shape=[1, 3, 512, 512], onnx_path='test.onnx')

# # 可视化特征图
# sam1 = pr_net.get_feature_maps(input_image=img, layer_name='encoder.conv1')
# fig = plt.figure(dpi=1000)
# for i in range(len(sam1)):
#     plt.subplot(8, 8, i + 1)
#     im = plt.imshow(sam1[i], cmap='jet')
#     plt.axis('off')
#
# fig.tight_layout()  # 调整整体空白
# plt.subplots_adjust(right=0.95, wspace=-0.5, hspace=0.1)  # 调整子图间距
# position = fig.add_axes([0.91, 0.12, 0.02, 0.78])  # 位置[左,下,右,上]
# fig.colorbar(im, cax=position)
#
# plt.show()

# 深度图到点云生成
# exr_depth_path = '/home/ch5225/chaohua/oaisys/oaisys_tmp/2022-02-24-17-27-51/batch_0002/sensorRight/0007sensorRight_pinhole_depth_00.exr'
# gt_depth = load_exr(exr_depth_path)
# pc = point_cloud_generator(focal_length=595.90, scalingfactor=1.0)
#
# pc.rgb = col_seg
# pc.depth = pr_depth
# pc.calculate()
# pc.write_ply('pc1.ply')
# pc.show_point_cloud()

# 可视化结果
# plt.subplot(221)
# plt.imshow(img)
# plt.axis('off')
# plt.subplot(222)
# plt.imshow(pr_seg)
# plt.axis('off')
# plt.subplot(223)
# plt.imshow(col_seg)
# plt.axis('off')
# plt.subplot(224)
# plt.imshow(pr_depth)
# plt.axis('off')
# plt.show()

# 读取mesh并进行可视化
# import open3d as o3d
#
# obj_path = '/home/ch5225/Downloads/downthemall/ZLF_0082_0674201499M818RAS_N0032430ZCAM03130_1100LUJ02.obj'
# mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
#
# rotate_mat = np.array([[1.0, 0.0, 0.0],
#                        [0.0, -1.0, 0.0],
#                        [0.0, 0.0, -1.0]])
#
# mesh = mesh.rotate(rotate_mat)
#
# print(np.asarray(mesh.vertices).shape)
# print(np.asarray(mesh.triangles).shape)
# texture = np.asarray(mesh.triangle_uvs)
# print("")
#
# print("Try to render a mesh with normals (exist: " +
#       str(mesh.has_vertex_normals()) + ") and colors (exist: " +
#       str(mesh.has_vertex_colors()) + ")")
# o3d.visualization.draw_geometries([mesh])

# 输出网络层的名字
# config_path = 'logs/2022_03_29_12_05_03/2022_03_29_12_05_03_config.yaml'
# model_weights_path = os.path.join(os.path.dirname(config_path), 'ep100.pth')
# print(model_weights_path)
# pr_net = PredictModel(config_path=config_path, model_weights_path=model_weights_path)
#
# module_names = []
# for name, layer in pr_net.net.named_modules():
#     module_names.append(name)
#
# print(module_names)

# 测试smp模块
# from torchsummary import summary
# import segmentation_models_pytorch as smp
#
# model = smp.create_model('Unet', encoder_name='resnet18', encoder_weights=None, in_channels=3, classes=2)
# summary(model.to('cuda'), (3, 512, 512))

# ******** 对模拟数据中的各类图片进行分类***********
# import glob
# import shutil
#
# #
# file_dir = '/media/ch5225/ch/oaisys-new'
#
# img_dir_list = glob.glob(file_dir + '/*/*/*Left', recursive=True)
#
# rgb_dir = file_dir + '/rgb'
# semantic_00_dir = file_dir + '/semantic_00'
# semantic_01_dir = file_dir + '/semantic_01'
# instance_dir = file_dir + '/instance'
# depth_exr_dir = file_dir + '/depth_exr'
#
# if not os.path.exists(rgb_dir): os.makedirs(rgb_dir)
# if not os.path.exists(semantic_00_dir): os.makedirs(semantic_00_dir)
# if not os.path.exists(semantic_01_dir): os.makedirs(semantic_01_dir)
# if not os.path.exists(instance_dir): os.makedirs(instance_dir)
# if not os.path.exists(depth_exr_dir): os.makedirs(depth_exr_dir)
#
# count = 10
#
# for img_dir in img_dir_list:
#
#     img_list = os.listdir(img_dir)
#
#     for img_name in img_list:
#         img = img_dir + '/' + img_name
#         if 'rgb' in img_name:
#             count += 1
#
#             rgb_img = img
#             instance_img = img.replace('rgb_00', 'instance_label_00')
#             exr_img = img.replace('rgb_00', 'pinhole_depth_00').replace('.png', '.exr')
#             s1_img = img.replace('rgb_00', 'semantic_label_00')
#             s2_img = img.replace('rgb_00', 'semantic_label_01')
#
#             shutil.move(rgb_img, rgb_dir + '/' + str(count).zfill(5) + 'Left.png')
#             shutil.move(rgb_img.replace('Left', 'Right'), rgb_dir + '/' + str(count).zfill(5) + 'Right.png')
#
#             shutil.move(s1_img, semantic_00_dir + '/' + str(count).zfill(5) + 'Left.png')
#             shutil.move(s1_img.replace('Left', 'Right'), semantic_00_dir + '/' + str(count).zfill(5) + 'Right.png')
#
#             shutil.move(s2_img, semantic_01_dir + '/' + str(count).zfill(5) + 'Left.png')
#             shutil.move(s2_img.replace('Left', 'Right'), semantic_01_dir + '/' + str(count).zfill(5) + 'Right.png')
#
#             shutil.move(instance_img, instance_dir + '/' + str(count).zfill(5) + 'Left.png')
#             shutil.move(instance_img.replace('Left', 'Right'), instance_dir + '/' + str(count).zfill(5) + 'Right.png')
#
#             shutil.move(exr_img, depth_exr_dir + '/' + str(count).zfill(5) + 'Left.exr')
#             shutil.move(exr_img.replace('Left', 'Right'), depth_exr_dir + '/' + str(count).zfill(5) + 'Right.exr')
#
#             if count % 100 == 0:
#                 print(count)

# 测试change color
# from utils.change_color_v1 import change_color_opencv
#
# change_color_opencv('/home/ch5225/Desktop/模拟数据/oaisys-new/rgb/00009Left.png', 'test.png',
#                     '/home/ch5225/Desktop/模拟数据/oaisys-new/sky=1/00009Left.png')


# 深度图变视差图
from predict_model import show_depth
from scipy import stats

exr_dir = '/home/ch5225/Desktop/模拟数据/oaisys-new/depth_exr/'
new_depth_dir = '/home/ch5225/Desktop/模拟数据/oaisys-new/disparity-npy/'
new_depth_png_dir = '/home/ch5225/Desktop/模拟数据/oaisys-new/disparity-png/'

if not os.path.exists(new_depth_dir): os.mkdir(new_depth_dir)
if not os.path.exists(new_depth_png_dir): os.mkdir(new_depth_png_dir)

count = 0
for exr_name in os.listdir(exr_dir)[45:]:
    # exr_path = '/home/ch5225/Desktop/模拟数据/oaisys-new/depth_exr/00151Left.exr'

    depth = cv2.imread(exr_dir+exr_name, cv2.IMREAD_UNCHANGED)
    depth = depth[:, :, 0]

    min_depth = np.min(depth)
    #
    depth[depth == 10000000000.00000] = -1 * min_depth
    max_depth = np.max(depth)
    # depth[depth == 0.0] = max_depth

    focal_length = 595.90
    base_line = 0.27

    disparity = focal_length * base_line / depth
    # mean_d = np.mean(disparity)
    # sigma = np.std(disparity)
    # disparity = (disparity - mean_d) / sigma

    min_d = np.min(np.abs(disparity))
    max_d = np.max(disparity)

    range = max_d - min_d
    disparity = (disparity-min_d)/range
    disparity[disparity < 0] = 0

    disparity_path = new_depth_dir + exr_name.replace('exr', 'npy')
    np.save(disparity_path, disparity)

    disparity_png_path = new_depth_png_dir + exr_name.replace('exr', 'png')
    d_img = show_depth(disparity)

    save_png(disparity_png_path, d_img)
    # save_png('test.png', d_img)
    count += 1
    if count % 100 == 0:
        print(count)

    # focal_length = 595.90
    # base_line = 0.27

    # disparity = focal_length * base_line / depth
    # disparity = cv2.normalize(disparity)

    # disparity1 = disparity-np.mean(disparity)/np.std(disparity)

# plt.imshow(disparity)
# plt.show()

# model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
#
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
#
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()
#
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
#
# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#     transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform
#
# img_path = '/home/ch5225/Desktop/模拟数据/oaisys-new/rgb/00150Left.png'
#
# # img_path = '/home/ch5225/chaohua/lunar_rocky_landscape/images/render_clean/render1164.png'
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# input_batch = transform(img).to(device)
#
# with torch.no_grad():
#     prediction = midas(input_batch)
#
#     prediction = torch.nn.functional.interpolate(
#         prediction.unsqueeze(1),
#         size=img.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()
#
# output = prediction.cpu().numpy()
#
# plt.imshow(output)
# plt.show()
''''''
