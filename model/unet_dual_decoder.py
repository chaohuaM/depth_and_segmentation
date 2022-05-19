# @Author  : ch
# @Time    : 2022/4/3 下午8:51
# @File    : unet_dual_decoder.py

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.modules import Conv2dReLU
from segmentation_models_pytorch.base.heads import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import CenterBlock


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.norm = nn.Sigmoid()

    def forward(self, inputs):
        mask = self.conv(inputs)
        mask = self.bn(mask)
        mask = self.norm(mask)

        return mask


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None, attention_map=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        if attention_map is not None:
            x = x * attention_map
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features, attention_maps=None):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        outputs = []
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            attention_map = attention_maps[i] if attention_maps else None
            x = decoder_block(x, skip, attention_map)
            outputs.append(x)

        return outputs


class UnetDualDecoder(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, encoder_name='resnet18',
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 with_spatial_attention=False):

        super(UnetDualDecoder, self).__init__()

        self.with_spatial_attention = with_spatial_attention

        self.encoder = smp.encoders.get_encoder(name=encoder_name, in_channels=in_channels)

        self.decoder1 = UnetDecoder(encoder_channels=self.encoder.out_channels,
                                    decoder_channels=decoder_channels)

        self.decoder2 = UnetDecoder(encoder_channels=self.encoder.out_channels,
                                    decoder_channels=decoder_channels)

        if self.with_spatial_attention:
            sa_blocks = [
                SpatialAttentionModule(in_ch) for in_ch in decoder_channels
            ]
            self.sa_blocks = nn.ModuleList(sa_blocks)

        self.final_output1 = SegmentationHead(in_channels=decoder_channels[-1],
                                              out_channels=num_classes)

        self.final_output2 = SegmentationHead(in_channels=decoder_channels[-1],
                                              out_channels=1)

    def forward(self, inputs):

        features = self.encoder(inputs)

        outputs2 = self.decoder2(*features)

        sa_maps = None

        if self.with_spatial_attention:
            sa_maps = []
            for i, sa_block in enumerate(self.sa_blocks):
                sa_map = sa_block(outputs2[i])
                sa_maps.append(sa_map)

        outputs1 = self.decoder1(*features, attention_maps=sa_maps)

        final1 = self.final_output1(outputs1[-1])

        final2 = self.final_output2(outputs2[-1])

        return final1, final2, sa_maps


def unet_dual_decoder(in_channels=3, num_classes=2, encoder_name='resnet18'):
    return UnetDualDecoder(in_channels, num_classes, encoder_name)


def unet_dual_decoder_with_sa(in_channels=3, num_classes=2, encoder_name='resnet18'):
    return UnetDualDecoder(in_channels, num_classes, encoder_name, with_spatial_attention=True)


if __name__ == '__main__':

    import torch
    from torchsummary import summary

    x = torch.zeros(2, 3, 512, 512)
    net = unet_dual_decoder_with_sa()
    net.eval()
    o1, o2, sa_maps = net(x)
    print(o1.shape)
    print(o2.shape)
    print(len(sa_maps))
    # summary(net.to('cuda'), (3, 512, 512))

    # for param in net.sa_blocks[3].conv.parameters():
    #     print(param)

