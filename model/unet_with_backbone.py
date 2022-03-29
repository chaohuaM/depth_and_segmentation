import torch
import torch.nn as nn

from model.resnet import *
from model.vgg import VGG16


class SpatialSEModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.norm = nn.Sigmoid()
        
    def forward(self, input):
        output = self.conv(input)
        sam_mask = self.norm(output)

        return sam_mask


class ChannelSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, in_layers):
        return in_layers * self.cSE(in_layers)


class UpBlocks(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpBlocks, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, sam=None):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        if sam is not None:
            outputs = outputs * sam
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class DecoderBranch(nn.Module):
    def __init__(self, in_filters, out_filters, is_sem=False):
        super(DecoderBranch, self).__init__()

        # upsampling
        # 64,64,512
        self.up_concat4 = UpBlocks(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = UpBlocks(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = UpBlocks(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = UpBlocks(in_filters[0], out_filters[0])

    def forward(self, feats, sams=None):
        [feat1, feat2, feat3, feat4, feat5] = feats

        if sams:
            [sam1, sam2, sam3, sam4] = sams
        else:
            sam1, sam2, sam3, sam4 = None, None, None, None
        up4 = self.up_concat4(feat4, feat5, sam4)
        up3 = self.up_concat3(feat3, up4, sam3)
        up2 = self.up_concat2(feat2, up3, sam2)
        up1 = self.up_concat1(feat1, up2, sam1)

        return up4, up3, up2, up1


class MyNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, pretrained=False, backbone='resnet18'):
        super(MyNet, self).__init__()

        self.backbone = backbone

        if self.backbone == 'vgg16':
            self.encoder = VGG16(in_channels=in_channels, pretrained=pretrained)
            en_filters = [192, 384, 768, 1024]
        elif self.backbone == "resnet50":
            self.encoder = resnet50(in_channels=in_channels, pretrained=pretrained)
            en_filters = [64, 256, 512, 1024, 2048]
        elif self.backbone == "resnet18":
            self.encoder = resnet18(in_channels=in_channels, pretrained=pretrained)
            en_filters = [64, 64, 128, 256, 512]
        elif self.backbone == "resnet34":
            self.encoder = resnet34(in_channels=in_channels, pretrained=pretrained)
            en_filters = [64, 64, 128, 256, 512]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(self.backbone))
        out_filters = [64, 128, 256, 512]
        
        in_filters = [out_filters[-3]+en_filters[0], out_filters[-2]+en_filters[1],
                      out_filters[-1]+en_filters[2], en_filters[3]+en_filters[4]]

        # upsampling
        # dual decoder branch
        self.sem_branch = DecoderBranch(in_filters, out_filters)

        self.sam1 = SpatialSEModule(out_filters[0])
        self.sam2 = SpatialSEModule(out_filters[1])
        self.sam3 = SpatialSEModule(out_filters[2])
        self.sam4 = SpatialSEModule(out_filters[3])

        self.depth_branch = DecoderBranch(in_filters, out_filters)

        if 'resnet' in self.backbone:
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final_sem = nn.Conv2d(out_filters[0], num_classes, 1)
        self.final_depth = nn.Conv2d(out_filters[0], 1, 1)

    def forward(self, inputs):

        feats = self.encoder.forward(inputs)

        depth_up4, depth_up3, depth_up2, depth_up1 = self.depth_branch.forward(feats)

        sam4 = self.sam4(depth_up4)
        sam3 = self.sam3(depth_up3)
        sam2 = self.sam2(depth_up2)
        sam1 = self.sam1(depth_up1)

        sams = [sam1, sam2, sam3, sam4]

        sem_up4, sem_up3, sem_up2, sem_up1 = self.sem_branch.forward(feats, sams)

        if self.up_conv is not None:
            sem_up1 = self.up_conv(sem_up1)

        final_sem = self.final_sem(sem_up1)

        if self.up_conv is not None:
            depth_up1 = self.up_conv(depth_up1)

        final_depth = self.final_depth(depth_up1)

        return final_sem, final_depth

    def freeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


class Unet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, pretrained=False, backbone='vgg16'):
        super(Unet, self).__init__()

        self.backbone = backbone

        if self.backbone == 'vgg16':
            self.vgg = VGG16(in_channels=in_channels, pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif self.backbone == "resnet50":
            self.resnet = resnet50(in_channels=in_channels, pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        elif self.backbone == "resnet18":
            self.resnet = resnet18(in_channels=in_channels, pretrained=pretrained)
            in_filters = [192, 320, 640, 768]
        elif self.backbone == "resnet34":
            self.resnet = resnet34(in_channels=in_channels, pretrained=pretrained)
            in_filters = [192, 320, 640, 768]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(self.backbone))
        out_filters = [64, 64, 64, 64]

        self.conv1 = nn.Conv2d(64, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1)

        # upsampling
        # 64,64,512
        self.up_sem_concat4 = UpBlocks(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_sem_concat3 = UpBlocks(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_sem_concat2 = UpBlocks(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_sem_concat1 = UpBlocks(in_filters[0], out_filters[0])

        self.up_depth_concat4 = UpBlocks(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_depth_concat3 = UpBlocks(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_depth_concat2 = UpBlocks(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_depth_concat1 = UpBlocks(in_filters[0], out_filters[0])

        if 'resnet' in self.backbone:
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final_sem = nn.Conv2d(out_filters[0], num_classes, 1)
        self.final_depth = nn.Conv2d(out_filters[0], 1, 1)

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif 'resnet' in self.backbone:
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        depth1 = self.conv1(feat1)
        depth2 = self.conv2(feat2)

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

        return final_sem, final_depth  # , depth1, depth2

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif 'resnet' in self.backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif 'resnet' in self.backbone:
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


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    x = torch.zeros(2, 3, 512, 512)
    net = MyNet(backbone='resnet50', pretrained=False)
    # model_resnet50 = resnet50(pretrained=False)
    y = net(x)
    for u in y:
        print(u.shape)
    summary(net.to('cuda'), (3, 512, 512))
