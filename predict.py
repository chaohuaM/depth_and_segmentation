import torch
import cv2
import numpy as np
import torch.nn.functional as F
from model.unet_with_backbone import Unet
from utils.utils import resize_and_centered, preprocess_input, normalization, load_exr
from depth2pointcloud import point_cloud_generator

model_path = 'logs/2022_03_09_01_16_20/ep100-losses1.479-val_loss1.887.pth'
in_channels = 3
model = Unet(backbone='resnet50', in_channels=in_channels, deformable_mode=False, pretrained=False)
input_shape = [256, 256]
model.load_state_dict(torch.load(model_path, map_location='cuda'))
model = model.eval()
model = model.cuda()

img_path = '/home/ch5225/chaohua/MarsData/Data/Rock-A/images/Rock-A/1516_1516MR0077180000204657E01_DXXX.jpg'
exr_path = '/home/ch5225/chaohua/oaisys/oaisys_tmp/2022-03-03-15-15-02/batch_0002/sensorLeft/0008sensorLeft_pinhole_depth_00.exr'

gt_depth = load_exr(exr_path)
img_raw = cv2.imread(img_path)
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

if in_channels == 1:
    img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
else:
    img = img_raw

original_h = img.shape[0]
original_w = img.shape[1]

input_img = resize_and_centered(img, input_shape)

if in_channels == 1:
    input_img = input_img[:, :, np.newaxis]
image_data = np.expand_dims(np.transpose(preprocess_input(np.array(input_img, np.float32)), (2, 0, 1)), 0)

with torch.no_grad():
    image_data = torch.from_numpy(image_data).cuda()
    seg_img, depth_img = model(image_data)

    pr_seg = seg_img[0]
    pr_depth = depth_img[0][0]

    pr_seg = F.softmax(pr_seg.permute(1, 2, 0), dim=-1).cpu().numpy()
    pr_seg = resize_and_centered(pr_seg, (original_h, original_w), reverse=True)

    pr_seg = pr_seg.argmax(axis=-1) * 255
    pr_seg[pr_seg > 125] = 225
    pr_seg[pr_seg < 125] = 50

    pr_seg = cv2.cvtColor(pr_seg.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    pr_depth = pr_depth.cpu().numpy()

    pr_depth[pr_depth <= 1e-4] = 0.0
    pr_depth = normalization(pr_depth) * 255

    pr_depth = resize_and_centered(pr_depth, (original_h, original_w), reverse=True)

# 深度图到点云生成
pc = point_cloud_generator(focal_length=2383.60, scalingfactor=1.0)

# pc.rgb = img_raw
# pc.depth = pr_depth
# pc.calculate()
# pc.write_ply('pc1.ply')
# pc.show_point_cloud()

cv2.imshow('seg', pr_seg.astype(np.uint8))
cv2.waitKey(100)
#
cv2.imshow('depth', pr_depth.astype(np.uint8))
cv2.waitKey()
