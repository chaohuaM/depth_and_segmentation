import os.path
import datetime
import argparse
from utils.dataloader import RockDataset
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import torch.optim as optim
from utils.utils_fit import fit_one_epoch
from model.create_model_nn import create_model
from utils.callbacks import LossHistory
import numpy as np


def save_options(data, path):
    import yaml
    with open(path, encoding='utf-8', mode='w') as f:
        try:
            yaml.dump(data=data, stream=f, allow_unicode=True)
        except Exception as e:
            print(e)


def add_model_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("MyModel")
    # 模型相关
    parser.add_argument('--model_name', default='unet', type=str,
                        help='model architecture', required=False)
    parser.add_argument('--backbone', default='resnet18', type=str,
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
    parser.add_argument('--depth_loss_factor', default=0.0, type=float, required=False,
                        help='')

    # 训练相关
    parser.add_argument('--epoch', default=100, type=int,
                        help='Epoch when freeze_train means freeze epoch', required=False)
    parser.add_argument('--batch_size', default=2, type=int, required=False,
                        help='freeze batch size of image in training')
    parser.add_argument('--lr', default=0.01, type=float, required=False,
                        help='learning rate of the optimizer')

    return parent_parser


def train_model():
    parser = argparse.ArgumentParser(description='Depth and Segmentation')

    parser = add_model_specific_args(parser)
    model_parser, _ = parser.parse_known_args()
    model_args = vars(model_parser)

    # 路径相关
    parser.add_argument('--timestamp', default=datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S'),
                        type=str, help="choose running mode, train or predict")
    parser.add_argument('--log_dir', type=str, help='log save directory', required=False,
                        default='logs/')
    parser.add_argument('--dataset_path', default='dataset/oaisys_data', type=str,
                        help='dataset_path', required=False)

    # 系统相关
    parser.add_argument('--seed', default=2022, help='pl training seed for reproducibility', required=False)
    parser.add_argument('--gpus', default=1, help='cuda availability', nargs="*", type=int, required=False)
    parser.add_argument('--gpu_bs', default=2, type=int, required=False,
                        help='the proper batch sizes for the gpu, set smaller number when OOM ')
    parser.add_argument('--num_workers', default=6, type=int, required=False)

    parser.add_argument('--mode', metavar='train or predict', default='train', type=str,
                        help="choose running mode, train or predict")

    args = parser.parse_args()

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
    lr = args.lr
    dice_loss = args.dice_loss
    focal_loss = args.focal_loss
    depth_loss_factor = args.depth_loss_factor
    # ------------------------------#
    #   输入图片的大小
    # ------------------------------#
    input_shape = args.input_shape
    in_channels = args.in_channels
    data_transform = args.data_transform
    batch_size = args.batch_size
    gpu_bs = args.gpu_bs
    accmulate_bs = int(batch_size / gpu_bs)
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
    cls_weights = np.array([1])

    timestamp = args.timestamp
    log_path = args.log_dir
    save_dir = os.path.join(log_path, model_name + '/' + timestamp)

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    config_path = os.path.join(save_dir, 'hparams.yaml')
    save_options(model_args, path=config_path)

    loss_history = LossHistory(save_dir)

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(os.path.join(dataset_path, "ImageSets/train.txt"), "r") as f:
        train_lines = f.readlines()

    with open(os.path.join(dataset_path, "ImageSets/val.txt"), "r") as f:
        val_lines = f.readlines()

    # 生成dataloader
    data_shape = [input_shape[0], input_shape[1], in_channels]

    train_dataset = RockDataset(train_lines, data_shape, num_classes, data_transform, dataset_path)
    val_dataset = RockDataset(val_lines, data_shape, num_classes, False, dataset_path)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=gpu_bs, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=gpu_bs, num_workers=num_workers,
                                pin_memory=True, drop_last=True)

    model = create_model(**model_args)
    model = model.cuda()
    model_train = model.train()

    print("-----------------*** start training ***--------------------------")
    epoch_step = len(train_lines) // batch_size
    epoch_step_val = len(val_lines) // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    optimizer = optim.Adam(model_train.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    metric_list = [0, 0, 0, 0, 0]

    for epoch_idx in range(epoch):
        metric = fit_one_epoch(model_train, loss_history, optimizer, epoch_idx,
                                 epoch_step, epoch_step_val, train_dataloader, val_dataloader, epoch, cuda=True,
                                 dice_loss=dice_loss, focal_loss=focal_loss, depth_loss_factor=depth_loss_factor)

        if metric >= min(metric_list):
            metric_list[metric_list.index(min(metric_list))] = metric
            # SAVE MODEL
            save_model_name = 'epoch={:0>3d}-miou={:.3f}.pth'.format(epoch_idx+1, metric)
            torch.save(model.state_dict(), os.path.join(loss_history.save_path,
                                                        save_model_name))

            print("best metric:", metric, "saved:", save_model_name)

        lr_scheduler.step()


if __name__ == "__main__":
    train_model()
