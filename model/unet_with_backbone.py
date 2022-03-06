import torch
import torch.nn as nn

from model.deformable_resnet import resnet50
from model.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, pretrained=False, backbone='vgg', deformable_mode=False):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(in_channels=3, pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(in_channels=3, pretrained=pretrained, deformable_mode=deformable_mode)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_sem_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_sem_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_sem_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_sem_concat1 = unetUp(in_filters[0], out_filters[0])

        self.up_depth_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_depth_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_depth_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_depth_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final_sem = nn.Conv2d(out_filters[0], num_classes, 1)
        self.final_depth = nn.Conv2d(out_filters[0], 1, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        sem_up4 = self.up_sem_concat4(feat4, feat5)
        sem_up3 = self.up_sem_concat3(feat3, sem_up4)
        sem_up2 = self.up_sem_concat2(feat2, sem_up3)
        sem_up1 = self.up_sem_concat1(feat1, sem_up2)

        if self.up_conv is not None:
            sem_up1 = self.up_conv(sem_up1)

        final_sem = self.final_sem(sem_up1)

        depth_up4 = self.up_depth_concat4(feat4, feat5)
        depth_up3 = self.up_depth_concat3(feat3, depth_up4)
        depth_up2 = self.up_depth_concat2(feat2, depth_up3)
        depth_up1 = self.up_depth_concat1(feat1, depth_up2)

        if self.up_conv is not None:
            depth_up1 = self.up_conv(depth_up1)

        final_depth = self.final_depth(depth_up1)

        return final_sem, final_depth

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)