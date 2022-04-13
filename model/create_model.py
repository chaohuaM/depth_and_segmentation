# @Author  : ch
# @Time    : 2022/4/3 下午1:58
# @File    : create_model.py

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from model.unet_dual_decoder import unet_dual_decoder, unet_dual_decoder_with_sa

from losses.depth_losses import BerHu_Loss
from utils.utils_metrics import binary_mean_iou
import torchmetrics


class MyModel(pl.LightningModule):
    def __init__(self, model_name, backbone, in_channels, num_classes, pretrained=None,
                 focal_loss=0, dice_loss=1, depth_loss_factor=1.0, **kwargs):
        """

        :param model_name:
        :param backbone:
        :param in_channels:
        :param num_classes:
        :param focal_loss:
        :param dice_loss:
        """
        super().__init__()

        self.model_name = model_name
        if self.model_name == 'unet_dual_decoder':
            self.model = unet_dual_decoder(in_channels=in_channels, num_classes=num_classes, encoder_name=backbone)
        elif self.model_name == 'unet_dual_decoder_with_sa':
            self.model = unet_dual_decoder_with_sa(in_channels=in_channels, num_classes=num_classes, encoder_name=backbone)
        else:
            self.model = smp.create_model(
                model_name, encoder_name=backbone, in_channels=in_channels, classes=num_classes, encoder_weights=pretrained)

        self.num_classes = num_classes
        self.focal_loss = focal_loss
        self.dice_loss = dice_loss
        self.depth_loss_factor = depth_loss_factor
        self.save_hyperparameters()

        self.sem_loss_fn = smp.losses.SoftBCEWithLogitsLoss()
        if self.dice_loss:
            self.sem_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
        if self.model_name in ['unet_dual_decoder', 'unet_dual_decoder_with_sa']:
            self.depth_loss = BerHu_Loss
        else:
            self.depth_loss = None

        self.f1_score = torchmetrics.F1Score(num_classes=num_classes, threshold=0.5)

    def forward(self, image):
        output = self.model(image)
        # TODO 找到一种简单的方式来判断是一个还是两个返回值
        return output

    def shared_step(self, batch, stage):
        images, masks, labels, depths = batch

        outputs = self.forward(images)

        outputs = list(outputs) if isinstance(outputs, tuple) else [outputs]

        sem_outputs = outputs[0]

        sem_loss = self.sem_loss_fn(sem_outputs, masks.float())

        self.log(f"{stage}_sem_loss", sem_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        depth_loss = 0

        if self.depth_loss is not None:
            depth_output = outputs[1]
            depth_loss = self.depth_loss_factor * self.depth_loss(depth_output, depths)
            self.log(f"{stage}_depth_loss", depth_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        total_loss = sem_loss + depth_loss

        self.f1_score.update(sem_outputs.view(-1), masks.view(-1))
        iou = binary_mean_iou(sem_outputs, labels)

        return {"loss": total_loss,
                "iou": iou
                }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        total_iou = 0

        for output in outputs:
            total_iou += output['iou']

        self.log(f"{stage}_f1_score", self.f1_score.compute(), logger=True)
        self.log(f"{stage}_iou", total_iou / len(outputs), logger=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MyModel")
        # 模型相关
        parser.add_argument('--model_name', default='deeplabv3plus', type=str,
                            help='model architecture', required=False)
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help='model backbone', required=False)
        parser.add_argument('--pretrained', default=None, type=str,
                            help='model backbone pretrained', required=False)
        # 输入数据相关
        parser.add_argument('--input_shape', default=[512, 512], nargs="*", type=int,
                            help='input image size [h, w]', required=False)
        parser.add_argument('--in_channels', default=3, type=int,
                            help='input image channels, 1 or 3', required=False)
        parser.add_argument('--num_classes', help='number of classes', default=1, required=False)
        parser.add_argument('--data_transform', type=int, default=0, required=False,
                            help='training data transform')

        # 损失函数相关
        parser.add_argument('--dice_loss', default=0, type=int, required=False)
        parser.add_argument('--focal_loss', default=0, type=int, required=False)
        parser.add_argument('--depth_loss_factor', default=1.0, type=float, required=False,
                            help='')

        # 训练相关
        parser.add_argument('--epoch', default=100, type=int,
                            help='Epoch when freeze_train means freeze epoch', required=False)
        parser.add_argument('--batch_size', default=16, type=int, required=False,
                            help='freeze batch size of image in training')
        parser.add_argument('--lr', default=0.01, type=float, required=False,
                            help='learning rate of the optimizer')

        return parent_parser


class TransferModel(MyModel):
    def __init__(self):
        super(TransferModel, self).__init__()
