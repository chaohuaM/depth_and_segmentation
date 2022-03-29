import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.optim as optim

# 测试dataloader函数
# from torch.utils.data import DataLoader
# from utils.dataloader import RockDataset, rock_dataset_collate
#
# train_lines = ['1306sensorRight_rgb_00', '4111sensorRight_rgb_00', '0752sensorLeft_rgb_00', '1080sensorLeft_rgb_00',
#                '3940sensorLeft_rgb_00', '4465sensorRight_rgb_00', '1665sensorRight_rgb_00', '3687sensorLeft_rgb_00',
#                '2669sensorLeft_rgb_00', '1702sensorLeft_rgb_00', '4359sensorLeft_rgb_00', '4550sensorRight_rgb_00']
# input_shape = [256, 256]
# input_channel = 1
# input_shape.append(input_channel)
# num_classes = 2
# data_dir = '/home/ch5225/Desktop/模拟数据/2022-02-02-00-23-59'
# train_dataset = RockDataset(train_lines, input_shape, num_classes, False, data_dir)
# gen = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=2, pin_memory=True,
#                  drop_last=True, collate_fn=rock_dataset_collate)
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


# import cv2
# import torch
#
# import matplotlib.pyplot as plt
#
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
# img = cv2.imread('/home/ch5225/chaohua/Rock-data/real_moon_image/train/TCAM15.png')
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
# plt.imshow(output)
# plt.show()

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

def save_png(img_path, data):
    cv2.imwrite(img_path, data)


# 将标签中的rgb值转为为0和1
import glob

# label_dir = '/home/ch5225/Desktop/模拟数据/2022-03-17-20-11-07/semantic_01/'
# new_label_dir = '/home/ch5225/Desktop/模拟数据/2022-03-17-20-11-07/semantic_01_label/'
#
# if not os.path.exists(new_label_dir): os.makedirs(new_label_dir)
#
# count = 0
#
# for img_path in glob.glob(label_dir + '*.png'):
#     img_name = img_path.split('/')[-1]
#     img = cv2.imread(img_path, 0)
#
#     img[img < 100] = 0
#     img[img > 100] = 1
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
from utils.utils import pth2onnx
from predict import PredictModel

config_path = 'logs/2022_03_27_21_48_23/2022_03_27_21_48_23_config.yaml'
pth_path = 'logs/2022_03_27_21_48_23/ep100.pth'
onnx_path = pth_path.replace('.pth', '.onnx')

net = PredictModel(config_path=config_path, model_weights_path=pth_path).net
pth2onnx(model=net, input_shape=[1, 3, 512, 512], onnx_path=onnx_path)

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
