import torch
import cv2
import numpy as np
import torch.nn.functional as F
from model.unet_with_backbone import Unet
from utils.dataloader import resize_image, preprocess_input


model_path = 'logs/ep050-loss1.118.pth'
model = Unet(backbone='resnet50', deformable_mode=False)
input_shape = [256, 256]
model.load_state_dict(torch.load(model_path, map_location='cuda'))
model = model.eval()
model = model.cuda()

img_path = '/home/ch5225/Desktop/模拟数据/2022-02-02-00-23-59/rgb/0117sensorRight_rgb_00.png'

img = cv2.imread(img_path, -1)
img = resize_image(img, input_shape)
image_data = np.expand_dims(np.transpose(preprocess_input(np.array(img, np.float32)), (2, 0, 1)), 0)


with torch.no_grad():
    image_data = torch.from_numpy(image_data).cuda()
    seg_img, depth_img = model(image_data)

    pr_seg = seg_img[0]
    pr_depth = depth_img[0][0]

    pr_seg = F.softmax(pr_seg.permute(1, 2, 0), dim=-1).cpu().numpy()
    pr_depth = pr_depth.cpu().numpy()


#
# cv2.imshow('rr', pr_depth)
# cv2.waitKey()



