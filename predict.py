import matplotlib.pyplot as plt
import yaml
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from model.unet_with_backbone import MyNet
from utils.utils import resize_and_centered, preprocess_input, normalization, load_exr
from depth2pointcloud import point_cloud_generator

colors = [(0, 0, 0), (0, 220, 85), (255, 23, 13), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
          (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
          (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
          (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]


def blend_image(raw, mask, blend_factor):
    col_mask = np.zeros((np.shape(mask)[0], np.shape(mask)[1], 3))
    for c in np.unique(mask):
        col_mask[mask == c] = colors[c]
    col_mask = cv2.addWeighted(raw, 1, col_mask.astype(np.uint8), blend_factor, 0)

    return col_mask


def show_depth(depth):
    depth = normalization(depth) * 255
    depth_img = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_MAGMA)

    return depth_img.astype(np.uint8)


class PredictModel(object):
    _defaults = {
        # -------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        # -------------------------------------------------------------------#
        "model_weights_path": '',
        # --------------------------------#
        #   所需要区分的类的个数+1
        # --------------------------------#
        "num_classes": 2,
        # --------------------------------#
        #   所使用的的主干网络：vgg、resnet50
        # --------------------------------#
        "backbone": "resnet50",
        # --------------------------------#
        #   输入图片的大小
        # --------------------------------#
        "in_channels": 3,
        "input_shape": [512, 512],
        # --------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # --------------------------------#
        "cuda": True,
    }

    # ---------------------------------------------------#
    #   初始化UNET
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        self.colors = colors
        # ---------------------------------------------------#
        #   获得模型
        # ---------------------------------------------------#
        if self.config_path:
            with open(self.config_path, 'r') as f:
                config_params = yaml.safe_load(f)
                for name, value in config_params.items():
                    setattr(self, name, value)
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        self.net = MyNet(num_classes=self.num_classes, backbone=self.backbone, in_channels=self.in_channels)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_weights_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_weights_path))

        if self.cuda:
            # self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        if self.in_channels == 1 and len(image.shape) != 1:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img = image
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        original_h = img.shape[0]
        original_w = img.shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        input_img = resize_and_centered(img, self.input_shape)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        if self.in_channels == 1:
            input_img = input_img[:, :, np.newaxis]
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(input_img, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            image_data = torch.from_numpy(image_data).cuda()
            seg_output, depth_output = self.net(image_data)

            pr_seg = seg_output[0]
            pr_depth = depth_output[0][0]

            pr_seg = F.softmax(pr_seg.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr_seg = resize_and_centered(pr_seg, (original_h, original_w), reverse=True)

            pr_depth = pr_depth.cpu().numpy()
            pr_depth = resize_and_centered(pr_depth, (original_h, original_w), reverse=True)

            pr_seg = pr_seg.argmax(axis=-1).astype(np.uint8)

        return pr_seg, pr_depth

    # 特征可视化，利用pytorch的 register_forward_hook() 函数
    def get_feature_maps(self, input_image, layer_name):
        """
        get feature maps by layer name,
        using the built-in pytorch hook function —— register_forward_hook()
        :param input_image: the inpput image to be detected, ndarray
        :param layer_name: the name of the layer to be visualized
        :return:  feature maps List[]
        """
        original_h = input_image.shape[0]
        original_w = input_image.shape[1]
        feature_maps = []
        def layer_hook(module, input, output):
            output = output[0].cpu().numpy()
            for feature in output:
                feature_maps.append(resize_and_centered(feature, (original_h, original_w), reverse=True))

        hook = self.net.get_submodule(layer_name).register_forward_hook(layer_hook)
        _, _ = self.detect_image(input_image)

        hook.remove()

        return feature_maps


if __name__ == '__main__':
    config_path = 'logs/2022_03_27_21_48_23/2022_03_27_21_48_23_config.yaml'
    model_weights_path = 'logs/2022_03_27_21_48_23/ep100.pth'
    pr_net = PredictModel(config_path=config_path, model_weights_path=model_weights_path)
    mode = 'dir_predict'
    image_path = '/home/ch5225/chaohua/MarsData/Data/Rock-B/images/349_0349ML0014190010108330E01_DXXX.jpg'
    # image_path = '/home/ch5225/Desktop/模拟数据/2022-02-02-00-23-59/rgb/0030sensorRight_rgb_00.png'
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pr_seg, pr_depth = pr_net.detect_image(img)
    # blend混合显示
    col_seg = blend_image(img, pr_seg, 0.3)

    col_depth = show_depth(pr_depth)

    cv2.imwrite('predicted_seg.png', cv2.cvtColor(col_seg, cv2.COLOR_RGBA2BGR))
    cv2.imwrite('predicted_depth.png', col_depth)

    cv2.imshow('seg', cv2.cvtColor(col_seg, cv2.COLOR_RGBA2BGR))
    cv2.waitKey(1000)
    cv2.imshow('depth', cv2.cvtColor(col_seg, cv2.COLOR_RGBA2BGR))
    cv2.waitKey()

