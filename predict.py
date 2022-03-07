import torch
import cv2
import numpy as np
import torch.nn.functional as F
from model.unet_with_backbone import Unet
from utils.utils import resize_and_centered, preprocess_input, normalization
from depth2pointcloud import point_cloud_generator


model_path = 'logs/ep100-losses0.602-val_loss0.695.pth'
model = Unet(backbone='resnet50', deformable_mode=False)
input_shape = [256, 256]
model.load_state_dict(torch.load(model_path, map_location='cuda'))
model = model.eval()
model = model.cuda()

img_path = '/home/ch5225/Desktop/模拟数据/2022-02-02-00-23-59/rgb/0107sensorRight_rgb_00.png'

img = cv2.imread(img_path)
original_h = img.shape[0]
original_w = img.shape[1]

input_img = resize_and_centered(img, input_shape)

image_data = np.expand_dims(np.transpose(preprocess_input(np.array(input_img, np.float32)), (2, 0, 1)), 0)


with torch.no_grad():
    image_data = torch.from_numpy(image_data).cuda()
    seg_img, depth_img = model(image_data)

    pr_seg = seg_img[0]
    pr_depth = depth_img[0][0]

    pr_seg = F.softmax(pr_seg.permute(1, 2, 0), dim=-1).cpu().numpy()
    pr_seg = resize_and_centered(pr_seg, (original_h, original_w), reversed=True)

    pr_seg = pr_seg.argmax(axis=-1) * 255

    pr_depth = pr_depth.cpu().numpy()

    pr_depth = normalization(pr_depth) * 255

    pr_depth = resize_and_centered(pr_depth, (original_h, original_w), reversed=True)


# 深度图到点云生成
# pc = point_cloud_generator(focal_length=13.11, scalingfactor=1.0)
#
# pc.rgb = img
# pc.depth = pr_depth
# pc.calculate()
# pc.write_ply('pc1.ply')
# pc.show_point_cloud()

# cv2.imshow('seg', pr_seg.astype(np.uint8))
# cv2.waitKey(100)
#
cv2.imshow('depth', pr_depth.astype(np.uint8))
cv2.waitKey()


