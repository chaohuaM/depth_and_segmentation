import os
import datetime

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model.create_model import MyModel
from utils.dataloader import RockDataset

import argparse
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

MODEL_NAME = ['Unet', 'DeepLab', 'unet_dual_decoder', 'unet_dual_decoder_with_sa']


def parse_argument():
    parser = argparse.ArgumentParser(description='Depth and Segmentation')
    # 路径相关
    parser.add_argument('--log_dir', type=str, help='log save directory', required=False,
                        default='logs/')
    parser.add_argument('--dataset_path', default='dataset/oaisys_data', type=str,
                        help='dataset_path', required=False)

    # 模型相关
    parser.add_argument('--model_name', default='unet_dual_decoder_with_sa', type=str,
                        help='model architecture', required=False)
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='model backbone', required=False)
    parser.add_argument('--pretrained', default=0, type=int,
                        help='model backbone pretrained', required=False)
    parser.add_argument('--model_path', default='', type=str,
                        help='pretrained model path', required=False)
    # 输入数据相关
    parser.add_argument('--input_shape', default=[512, 512], nargs="*", type=int,
                        help='input image size [h, w]', required=False)
    parser.add_argument('--in_channels', default=3, type=int,
                        help='input image channels, 1 or 3', required=False)
    parser.add_argument('--num_classes', help='number of classes', default=1, required=False)
    parser.add_argument('--transform', type=int, default=0, required=False,
                        help='training data transform')
    # 训练相关
    parser.add_argument('--epoch', default=50, type=int,
                        help='Epoch when freeze_train means freeze epoch', required=False)
    parser.add_argument('--batch_size', default=1, type=int, required=False,
                        help='freeze batch size of image in training')

    # 损失函数相关
    parser.add_argument('--dice_loss', default=0, type=int, required=False)
    parser.add_argument('--focal_loss', default=0, type=int, required=False)
    parser.add_argument('--cls_weights', default="", type=str, required=False)

    # 系统相关
    parser.add_argument('--seed', default=2022, help='pl training seed for reproducibility', required=False)
    parser.add_argument('--gpus', default=1, help='cuda availability', nargs="*", type=int, required=False)
    parser.add_argument('--num_workers', default=8, type=int, required=False)

    parser.add_argument('--mode', metavar='train or predict', default='train', type=str,
                        help="choose running mode, train or predict")
    parser.add_argument('--timestamp', default=datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S'),
                        type=str, help="choose running mode, train or predict")

    # 变成对象后修改
    args = parser.parse_args()

    return args


def train_model(args):
    pl.seed_everything(args.seed)
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    gpus = args.gpus
    epoch = args.epoch
    # -------------------------------#
    #   训练自己的数据集必须要修改的
    #   自己需要的分类个数+1，如2 + 1
    # -------------------------------#
    num_classes = args.num_classes
    # -------------------------------#
    #   主干网络选择
    #   vgg、resnet50、resnet18
    # -------------------------------#
    model_name = args.model_name
    backbone = args.backbone
    model_path = args.model_path
    # ------------------------------#
    #   输入图片的大小
    # ------------------------------#
    input_shape = args.input_shape
    in_channels = args.in_channels
    transform = args.transform
    batch_size = args.batch_size
    # ------------------------------#
    #   数据集路径
    # ------------------------------#
    dataset_path = args.dataset_path
    # ---------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    # ---------------------------------------------------------------------#
    dice_loss = args.dice_loss
    # ---------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    # ---------------------------------------------------------------------#
    focal_loss = args.focal_loss
    # ------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------#
    num_workers = args.num_workers

    timestamp = args.timestamp
    log_path = args.log_dir
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_path, name=model_name, version=timestamp)
    csv_logger = pl_loggers.CSVLogger(save_dir=log_path, name=model_name,  version=timestamp)

    input_shape.append(in_channels)

    model = MyModel(model_name=model_name, backbone=backbone, in_channels=in_channels, num_classes=num_classes)

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(os.path.join(dataset_path, "ImageSets/train.txt"), "r") as f:
        train_lines = f.readlines()

    with open(os.path.join(dataset_path, "ImageSets/val.txt"), "r") as f:
        val_lines = f.readlines()

    train_dataset = RockDataset(train_lines, input_shape, num_classes, transform, dataset_path)
    val_dataset = RockDataset(val_lines, input_shape, num_classes, False, dataset_path)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=True, drop_last=True)

    trainer = pl.Trainer(gpus=gpus, max_epochs=epoch,
                         accumulate_grad_batches=16,
                         logger=[tb_logger, csv_logger],
                         default_root_dir=log_path)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )


if __name__ == '__main__':
    options = parse_argument()
    train_model(options)
