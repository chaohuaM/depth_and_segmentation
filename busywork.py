import cv2
import numpy as np
import torch
import torch.optim as optim

# 测试dataloader函数
from torch.utils.data import DataLoader
from utils.dataloader import RockDataset, rock_dataset_collate

train_lines = ['1306sensorRight_rgb_00', '4111sensorRight_rgb_00', '0752sensorLeft_rgb_00', '1080sensorLeft_rgb_00',
               '3940sensorLeft_rgb_00', '4465sensorRight_rgb_00', '1665sensorRight_rgb_00', '3687sensorLeft_rgb_00',
               '2669sensorLeft_rgb_00', '1702sensorLeft_rgb_00', '4359sensorLeft_rgb_00', '4550sensorRight_rgb_00']
input_shape = [256, 256]
num_classes = 2
data_dir = '/home/ch5225/Desktop/模拟数据/2022-02-02-00-23-59'
train_dataset = RockDataset(train_lines, input_shape, num_classes, False, data_dir)
gen = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=2, pin_memory=True,
                 drop_last=True, collate_fn=rock_dataset_collate)

# for batch in gen:
#     depths = torch.from_numpy(batch[3]).type(torch.FloatTensor)
#     depths = torch.unsqueeze(depths, 1)

# 测试gradient difference loss
# from losses.grad_loss import GRAD_LOSS, imgrad_yx
#
# loss1 = imgrad_yx(depths)
# loss = GRAD_LOSS(depths, depths).item()

# 测试模型
from model.unet_with_backbone import Unet


model = Unet(backbone='resnet50', deformable_mode=False).cuda()
# x = torch.zeros(2, 3, 512, 512).cuda()
# y, y_depth = model(x)

model_train = model.train()
# 测试是否可以训练
from utils.callbacks import LossHistory
from utils.utils_fit import fit_one_epoch_no_val

batch_size = 2
cls_weights = np.ones([num_classes], np.float32)
loss_history = LossHistory("logs/")
epoch_step = len(train_lines) // batch_size

optimizer = optim.Adam(model_train.parameters(), 1e-4)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

for epoch in range(0, 50):
    fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, 50, True, False,
                         False, cls_weights, num_classes)
    lr_scheduler.step()


