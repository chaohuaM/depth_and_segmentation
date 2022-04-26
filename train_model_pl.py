import os
import datetime

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model.create_model_pl import MyModel
from utils.dataloader import RockDataset

import argparse
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


MODEL_NAME = ['unet', 'deeplabv3plus', 'unet_dual_decoder', 'unet_dual_decoder_with_sa']


def train_model():
    parser = argparse.ArgumentParser(description='Depth and Segmentation')

    parser = MyModel.add_model_specific_args(parser)
    model_parser, _ = parser.parse_known_args()
    model_args = vars(model_parser)

    # 路径相关
    parser.add_argument('--timestamp', default=datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S'),
                        type=str, help="choose running mode, train or predict")
    parser.add_argument('--log_dir', type=str, help='log save directory', required=False,
                        default='new-logs/')
    parser.add_argument('--dataset_path', default='dataset/oaisys-new', type=str,
                        help='dataset_path', required=False)

    # 系统相关
    parser.add_argument('--seed', default=2022, help='pl training seed for reproducibility', required=False)
    parser.add_argument('--gpus', default=1, help='cuda availability', nargs="*", type=int, required=False)
    parser.add_argument('--gpu_bs', default=8, type=int, required=False,
                        help='the proper batch sizes for the gpu, set smaller number when OOM ')
    parser.add_argument('--num_workers', default=6, type=int, required=False)

    parser.add_argument('--mode', metavar='train or predict', default='train', type=str,
                        help="choose running mode, train or predict")

    args = parser.parse_args()

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
    model_name = args.model_name.lower()
    backbone = args.backbone
    # ------------------------------#
    #   输入图片的大小
    # ------------------------------#
    input_shape = args.input_shape
    in_channels = args.in_channels
    data_transform = args.data_transform
    batch_size = args.batch_size
    gpu_bs = args.gpu_bs
    accmulate_bs = int(batch_size/gpu_bs)
    # ------------------------------#
    #   数据集路径
    # ------------------------------#
    dataset_path = args.dataset_path
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

    ckpt_callback = ModelCheckpoint(save_top_k=5,
                                    monitor="val_f1_score",
                                    mode="max",
                                    filename="{epoch}-{val_fscore:.3f}")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    data_shape = [input_shape[0], input_shape[1], in_channels]

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(os.path.join(dataset_path, "ImageSets/train.txt"), "r") as f:
        train_lines = f.readlines()

    with open(os.path.join(dataset_path, "ImageSets/val.txt"), "r") as f:
        val_lines = f.readlines()

    train_dataset = RockDataset(train_lines, data_shape, num_classes, data_transform, dataset_path)
    val_dataset = RockDataset(val_lines, data_shape, num_classes, False, dataset_path)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=gpu_bs, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=gpu_bs, num_workers=num_workers,
                                pin_memory=True, drop_last=True)

    model = MyModel(**model_args)
    print("model set up with following hyper parameters:")
    print(model.hparams)

    trainer = pl.Trainer(default_root_dir=log_path,
                         gpus=gpus, max_epochs=epoch,
                         accumulate_grad_batches=accmulate_bs,
                         reload_dataloaders_every_n_epochs=5,
                         logger=[tb_logger, csv_logger],
                         callbacks=[ckpt_callback, lr_monitor])

    print("---------------- start training --------------")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader)

    print(f"best model path:{ckpt_callback.best_model_path}, score:{ckpt_callback.best_model_score}")


if __name__ == '__main__':

    train_model()
