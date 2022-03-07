import cv2
import numpy as np
import torch
import torch.optim as optim

# # 测试dataloader函数
# from torch.utils.data import DataLoader
# from utils.dataloader import RockDataset, rock_dataset_collate
#
# train_lines = ['1306sensorRight_rgb_00', '4111sensorRight_rgb_00', '0752sensorLeft_rgb_00', '1080sensorLeft_rgb_00',
#                '3940sensorLeft_rgb_00', '4465sensorRight_rgb_00', '1665sensorRight_rgb_00', '3687sensorLeft_rgb_00',
#                '2669sensorLeft_rgb_00', '1702sensorLeft_rgb_00', '4359sensorLeft_rgb_00', '4550sensorRight_rgb_00']
# input_shape = [256, 256]
# num_classes = 2
# data_dir = '/home/ch5225/Desktop/模拟数据/2022-02-02-00-23-59'
# train_dataset = RockDataset(train_lines, input_shape, num_classes, False, data_dir)
# gen = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=2, pin_memory=True,
#                  drop_last=True, collate_fn=rock_dataset_collate)

# for batch in gen:
#     depths = torch.from_numpy(batch[3]).type(torch.FloatTensor)
#     depths = torch.unsqueeze(depths, 1)

# 测试gradient difference loss
# from losses.grad_loss import GRAD_LOSS, imgrad_yx
#
# loss1 = imgrad_yx(depths)
# loss = GRAD_LOSS(depths, depths).item()

# # 测试模型
# from model.unet_with_backbone import Unet
#
#
# model = Unet(backbone='resnet50', deformable_mode=False).cuda()
# # x = torch.zeros(2, 3, 512, 512).cuda()
# # y, y_depth = model(x)
#
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

# 查看深度图
from utils.utils import load_exr

left_exr_path = '/home/ch5225/Desktop/模拟数据/2022-02-02-00-23-59/depth_exr/0100sensorLeft_pinhole_depth_00.exr'
right_exr_path = '/home/ch5225/Desktop/模拟数据/2022-02-02-00-23-59/depth_exr/0100sensorRight_pinhole_depth_00.exr'

left_depth = load_exr(left_exr_path)
right_depth = load_exr(right_exr_path)

# np.seterr(divide='ignore', invalid='ignore')
# left_disparity = 2382.82 * 270 / (left_depth * 1000)
# right_disparity = 2382.82 * 270 / (right_depth * 1000)


# import torch.onnx
# from model.unet_with_backbone import Unet
#
# batch_size = 1
# model_path = 'logs/ep100-losses0.602-val_loss0.695.pth'
# model = Unet(backbone='resnet50', deformable_mode=False)
# input_shape = [256, 256]
# model.load_state_dict(torch.load(model_path, map_location='cuda'))
# model = model.eval()
# # model = model.cuda()
#
# x = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
# torch_out = model(x)
#
# # Export the model
# torch.onnx.export(model,  # model being run
#                   x,  # model input (or a tuple for multiple inputs)
#                   "depth_and_segmentation.onnx",  # where to save the model (can be a file or file-like object)
#                   export_params=True,  # store the trained parameter weights inside the model file
#                   opset_version=11,  # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names=['input'],  # the model's input names
#                   output_names=['output'],  # the model's output names
#                   dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
#                                 'output': {0: 'batch_size'}})
