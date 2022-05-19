# @Author  : ch
# @Time    : 2022/4/3 下午1:58
# @File    : create_model_pl.py

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from model.unet_dual_decoder import unet_dual_decoder, unet_dual_decoder_with_sa

from losses.depth_losses import BerHu_Loss
from losses.ssi_loss import ScaleAndShiftInvariantLoss
from utils.utils_metrics import binary_mean_iou
import torchmetrics

MODEL_NAME = ['unet', 'deeplabv3plus', 'unet_dual_decoder', 'unet_dual_decoder_with_sa']
SEM_LOSS_FN = ['ce_loss', 'dice_loss']
DEPTH_LOSS_FN = ['berhu_loss', 'ssi_loss']


class MyModel(pl.LightningModule):
    def __init__(self, model_name, backbone, in_channels, num_classes, pretrained=None,
                 sem_loss_fn='ce_loss', depth_loss_fn='berhu_loss', depth_loss_factor=1.0,
                 use_depth_mask=True, **kwargs):
        """

        :param model_name:
        :param backbone:
        :param in_channels:
        :param num_classes:
        :param pretrained:
        :param sem_loss_fn:
        :param depth_loss_fn:
        :param depth_loss_factor:
        :param kwargs:
        """
        super().__init__()

        model_name = model_name.lower()
        sem_loss_fn = sem_loss_fn.lower()
        depth_loss_fn = depth_loss_fn.lower()

        assert model_name in MODEL_NAME, f"{model_name} is not in present supported model list:{MODEL_NAME}"
        assert sem_loss_fn in SEM_LOSS_FN, f"{sem_loss_fn} is not in present supported semantic loss function list:{SEM_LOSS_FN}"
        assert depth_loss_fn in DEPTH_LOSS_FN, f"{depth_loss_fn} is not in present supported depth loss function list:{DEPTH_LOSS_FN}"

        self.model_name = model_name
        if self.model_name == 'unet_dual_decoder':
            self.model = unet_dual_decoder(in_channels=in_channels, num_classes=num_classes, encoder_name=backbone)
        elif self.model_name == 'unet_dual_decoder_with_sa':
            self.model = unet_dual_decoder_with_sa(in_channels=in_channels, num_classes=num_classes,
                                                   encoder_name=backbone)
        else:
            self.model = smp.create_model(
                model_name, encoder_name=backbone, in_channels=in_channels, classes=num_classes,
                encoder_weights=pretrained)

        self.num_classes = num_classes

        self.depth_loss_factor = depth_loss_factor
        self.use_depth_mask = use_depth_mask

        self.sem_loss_fn = sem_loss_fn
        self.depth_loss_fn = depth_loss_fn
        self.save_hyperparameters()

        if self.sem_loss_fn == 'ce_loss':
            if self.num_classes == 1:   # 二类分类情况下采用bce-loss
                self.sem_loss = smp.losses.SoftBCEWithLogitsLoss()
            else:                       # 多类别时使用ce-loss
                self.sem_loss = smp.losses.SoftCrossEntropyLoss()
        elif self.sem_loss_fn == 'dice_loss':
            self.sem_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)

        if self.model_name in ['unet_dual_decoder', 'unet_dual_decoder_with_sa']:
            if self.depth_loss_fn == 'berhu_loss':
                self.depth_loss = BerHu_Loss
            elif self.depth_loss_fn == 'ssi_loss':
                self.depth_loss = ScaleAndShiftInvariantLoss()
        else:
            self.depth_loss = None

        self.f1_score = torchmetrics.F1Score(num_classes=num_classes, threshold=0.5)
        # self.iou = torchmetrics.IoU(num_classes=num_classes+1, threshold=0.5)
        self.iou = torchmetrics.JaccardIndex(num_classes=num_classes + 1, threshold=0.5)

    def forward(self, image):
        output = self.model(image)
        # TODO 找到一种简单的方式来判断是一个还是两个返回值
        return output

    def on_train_start(self) -> None:
        # log hyperparams
        self.logger.log_hyperparams(self.hparams,
                                    {'val_f1_score': 0, 'train_f1_score': 0, 'val_iou': 0, 'train_iou': 0})
        return super().on_train_start()

    def shared_step(self, batch, batch_idx, stage):
        images, masks, depths = batch

        outputs = self.forward(images)

        outputs = list(outputs) if isinstance(outputs, tuple) else [outputs]

        sem_outputs = outputs[0]

        sem_loss = self.sem_loss(sem_outputs, masks.float())

        self.log(f"{stage}_sem_loss", sem_loss, prog_bar=True, logger=True, on_epoch=True)

        depth_loss = 0

        if self.depth_loss is not None:
            depth_output = outputs[1]
            depth_output = depth_output.squeeze(1)
            depths = depths.squeeze(1)
            depth_masks = torch.ones_like(depths)
            depth_masks = depth_masks.type_as(depths)
            if self.use_depth_mask:
                depth_masks[torch.where(depths == 0)] = 0

            depth_loss = self.depth_loss_factor * self.depth_loss(depth_output, depths, depth_masks)
            self.log(f"{stage}_depth_loss", depth_loss, prog_bar=True, logger=True, on_epoch=True)

        total_loss = sem_loss + depth_loss

        if 'sa' in self.model_name:
            if stage == 'val' and batch_idx == 1:
                sa_maps = outputs[2]
                for idx, sa_map in enumerate(sa_maps):
                    self.logger.experiment.add_image(str(batch_idx)+"sa-map-"+str(idx), sa_map[0], global_step=self.global_step)
                    fig = plt.figure()
                    feature_map = sa_map[0][0].cpu().numpy()
                    plt.imshow(feature_map, cmap='jet', vmax=1.0, vmin=0.0)
                    plt.colorbar()
                    # plt.axis('off')
                    self.logger.experiment.add_figure(str(batch_idx)+"sa-map-f-"+str(idx), fig, global_step=self.global_step)

        self.f1_score(sem_outputs.view(-1), masks.view(-1))
        self.iou(sem_outputs.view(-1), masks.view(-1))

        self.log(f"{stage}_f1_score", self.f1_score, logger=True)
        self.log(f"{stage}_iou", self.iou, logger=True)

        return total_loss

    def shared_epoch_end(self, outputs, stage):

        # self.f1_score.compute()
        # self.iou.compute()
        #
        # self.log(f"{stage}_f1_score", self.f1_score, logger=True)
        # self.log(f"{stage}_iou", self.iou, logger=True)
        #
        # self.iou.reset()
        # self.f1_score.reset()
        pass

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")

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
        parser.add_argument('--model_name', default='unet', type=str,
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
        parser.add_argument('--sem_loss_fn', default='ce_loss', type=str, required=False)
        parser.add_argument('--depth_loss_fn', default='berhu_loss', type=str, required=False)
        parser.add_argument('--use_depth_mask', type=int, default=1, required=False)
        parser.add_argument('--depth_loss_factor', default=1.0, type=float, required=False,
                            help='the weight factor of the depth loss component')

        # 训练相关
        parser.add_argument('--epoch', default=100, type=int,
                            help='Epoch when freeze_train means freeze epoch', required=False)
        parser.add_argument('--batch_size', default=16, type=int, required=False,
                            help='freeze batch size of image in training')
        parser.add_argument('--lr', default=0.1, type=float, required=False,
                            help='learning rate of the optimizer')

        return parent_parser


class TransferModelPL(MyModel):
    def __init__(self, model_name, backbone, in_channels, num_classes):
        super(TransferModelPL, self).__init__(model_name, backbone, in_channels, num_classes)
        self.model.encoder.freeze()


if __name__ == '__main__':
    # model = MyModel(model_name='unet', backbone='resnet18', in_channels=1, num_classes=9)
    # print(model.metric)
    # x = torch.zeros(2, 3, 512, 512)
    #
    # y = model.model(x)
    # for u in y:
    #     print(u.shape)

    # tl_unet = TransferModel()
    model = TransferModelPL(model_name='unet', backbone='resnet18', in_channels=1, num_classes=9)
