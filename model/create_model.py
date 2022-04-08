# @Author  : ch
# @Time    : 2022/4/3 下午1:58
# @File    : create_model.py

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from model.unet_dual_decoder import unet_dual_decoder, unet_dual_decoder_with_sa
from losses.semantic_losses import Focal_Loss, CE_Loss, Dice_loss
from losses.depth_losses import BerHu_Loss
from utils.utils_metrics import f_score, binary_mean_iou

# smp.losses.DiceLoss()
class MyModel(pl.LightningModule):
    def __init__(self, model_name, backbone, in_channels, num_classes, focal_loss=0, dice_loss=1):
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
                model_name, encoder_name=backbone, in_channels=in_channels, classes=num_classes)

        self.num_classes = num_classes
        self.focal_loss = focal_loss
        self.dice_loss = dice_loss
        self.save_hyperparameters()

        self.sem_loss_fn = smp.losses.SoftBCEWithLogitsLoss()
        if self.dice_loss:
            self.sem_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
        if self.model_name in ['unet_dual_decoder', 'unet_dual_decoder_with_sa']:
            self.depth_loss = BerHu_Loss
        else:
            self.depth_loss = None

    def forward(self, image):
        output = self.model(image)
        # TODO 找到一种简单的方式来判断是一个还是两个返回值
        return output

    def shared_step(self, batch, stage):
        images, masks, labels, depths = batch

        outputs = self.forward(images)

        outputs = list(outputs) if isinstance(outputs, tuple) else [outputs]

        sem_outputs = outputs[0]

        sem_loss = self.sem_loss_fn(sem_outputs, masks)

        self.log(f"{stage}_sem_loss", sem_loss, prog_bar=True, logger=True)

        depth_loss = 0

        if self.depth_loss is not None:
            depth_output = outputs[1]
            depth_loss = self.depth_loss(depth_output, depths)
            self.log(f"{stage}_depth_loss", depth_loss, prog_bar=True, logger=True)

        total_loss = sem_loss + depth_loss

        _f_score = f_score(sem_outputs, labels)
        iou = binary_mean_iou(sem_outputs, labels)

        return {"loss": total_loss,
                "sem_loss": sem_loss,
                "depth_loss": depth_loss,
                "f_score": _f_score,
                "iou": iou
                }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        total_f_score = 0
        total_iou = 0

        for output in outputs:
            total_f_score += output['f_score']
            total_iou += output['iou']

        self.log(f"{stage}_f_score", total_f_score/len(outputs), logger=True)
        self.log(f"{stage}_iou", total_iou / len(outputs), logger=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

