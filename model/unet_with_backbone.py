import torch
import torch.nn as nn

from model.resnet import *
from model.vgg import VGG16


class DepthSpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.norm = nn.Sigmoid()
        
    def forward(self, input):
        output = self.conv(input)
        dsa_mask = self.norm(output)

        return dsa_mask


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_size, out_size, with_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        if with_bn:
            self.bn = nn.BatchNorm2d(out_size)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.bn is not None:
            outputs = self.bn(outputs)
        outputs = self.relu(outputs)

        return outputs


class UpBlocks(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpBlocks, self).__init__()
        self.conv1 = ConvBnRelu2d(in_size, out_size)
        self.conv2 = ConvBnRelu2d(out_size, out_size)

    def forward(self, inputs1, inputs2, sam=None):
        inputs = torch.cat([inputs1, self.up(inputs2)], 1)
        if sam is not None:
            inputs = inputs * sam
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        return outputs


class DecoderBranch(nn.Module):
    def __init__(self, in_filters, out_filters):
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
        self.depth_branch = DecoderBranch(in_filters, out_filters)

        self.sam1 = DepthSpatialAttentionModule(out_filters[0])
        self.sam2 = DepthSpatialAttentionModule(out_filters[1])
        self.sam3 = DepthSpatialAttentionModule(out_filters[2])
        self.sam4 = DepthSpatialAttentionModule(out_filters[3])

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


def weights_init(net, init_type='kaiming', init_gain=0.02):
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
    net = MyNet(backbone='resnet18', pretrained=False)
    # model_resnet50 = resnet50(pretrained=False)
    y = net(x)
    for u in y:
        print(u.shape)
    summary(net.to('cuda'), (3, 512, 512))
