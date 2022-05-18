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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs):
        outputs = self.net(inputs)

        return outputs

    def set_model(self, model):
        self.net = model

    def get_model(self):
        return self.net

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
    model = create_model()
    model_weights_path = ''
    TransferModel = MyModel()
    TransferModel.set_model(model)
    TransferModel.load_model_weights(model_weights_path)
    TransferModel.freeze_backbone()

    transfer_model = TransferModel.get_model()

    for param in transfer_model.encoder.parameters():
        print(param.requires_grad)

    for param in model.encoder.parameters():
        print(param.requires_grad)

    print(id(transfer_model), id(model))








