# @Author  : ch
# @Time    : 2022/4/17 下午1:19
# @File    : create_model_nn.py

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from model.unet_dual_decoder import unet_dual_decoder, unet_dual_decoder_with_sa


MODEL_NAME = ['unet', 'deeplabv3plus', 'unet_dual_decoder', 'unet_dual_decoder_with_sa']


class MyModel(nn.Module):
    def __init__(self, model_name='unet', backbone='resnet18', in_channels=3,
                 num_classes=1, pretrained=None, **kwargs):
        super().__init__()
        self.net = create_model(model_name, backbone, in_channels,
                 num_classes, pretrained, **kwargs)

    def forward(self, inputs):
        outputs = self.net(inputs)

        return outputs

    def freeze_backbone(self):
        for param in self.net.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.net.encoder.parameters():
            param.requires_grad = True


def create_model(model_name='unet', backbone='resnet18', in_channels=3,
                 num_classes=1, pretrained=None, **kwargs):
    # 判断是否在目前支持的模型列表中
    model_name = model_name.lower()
    assert model_name in MODEL_NAME, f"{model_name} is not in present supported model list:{MODEL_NAME}"

    # 按名称创建模型
    if model_name == 'unet_dual_decoder':
        model = unet_dual_decoder(in_channels=in_channels, num_classes=num_classes, encoder_name=backbone)
    elif model_name == 'unet_dual_decoder_with_sa':
        model = unet_dual_decoder_with_sa(in_channels=in_channels, num_classes=num_classes, encoder_name=backbone)
    else:
        model = smp.create_model(
            arch=model_name, encoder_name=backbone, in_channels=in_channels, classes=num_classes,
            encoder_weights=pretrained)

    return model


if __name__ == "__main__":
    model = MyModel()
    net = model.net

    for param in net.encoder.parameters():
        print(param.requires_grad)

    model.freeze_backbone()

    for param in net.encoder.parameters():
        print(param.requires_grad)

    print(id(net), id(model.net))








