# @Author  : ch
# @Time    : 2022/4/25 下午4:11
# @File    : depth_losses.py

import torch
import torch.nn as nn


import torch
import torch.nn as nn

from losses.AutomaticWeightedLoss import AutomaticWeightedLoss


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def berhu_loss(predict, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    x = predict - target
    abs_x = torch.abs(x)
    c = torch.max(abs_x).item() / 5.0
    leq = (abs_x <= c).float()
    l2_losses = (x ** 2 + c ** 2) / (2 * c)
    image_loss = leq * abs_x + (1 - leq) * l2_losses
    return reduction(image_loss * mask, M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.data_loss = MSELoss(reduction=reduction)
        self.regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.alpha = alpha
        self.awl = AutomaticWeightedLoss(2)

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        """
        :param prediction:  N ( x C) x H x W
        :param target:      N ( x C) x H x W
        :param mask:        N ( x C) x H x W
        :return:
        """
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        # total = self.__data_loss(self.__prediction_ssi, target, mask)
        # if self.__alpha > 0:
        #     total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        data_loss = self.data_loss(self.__prediction_ssi, target, mask)
        regularization_loss = self.regularization_loss(self.__prediction_ssi, target, mask)
        total = self.awl(data_loss, regularization_loss)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class BerHuLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.data_loss = berhu_loss
        self.regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.alpha = alpha
        self.awl = AutomaticWeightedLoss(2)

    def forward(self, prediction, target, mask):
        """
        :param prediction:  N ( x C) x H x W
        :param target:      N ( x C) x H x W
        :param mask:        N ( x C) x H x W
        :return:
        """
        # total = self.__data_loss(prediction, target, mask)
        # if self.__alpha > 0:
        #     total += self.__alpha * self.__regularization_loss(prediction, target, mask)

        data_loss = self.data_loss(prediction, target, mask)
        regularization_loss = self.regularization_loss(prediction, target, mask)
        total = self.awl(data_loss, regularization_loss)

        return total


if __name__ == '__main__':

    grad_smooth_factor0 = 0.0
    grad_smooth_factor1 = 1.0
    ssi_loss0 = ScaleAndShiftInvariantLoss(alpha=grad_smooth_factor0)
    ssi_loss1 = ScaleAndShiftInvariantLoss(alpha=grad_smooth_factor1)
    berhu_loss0 = BerHuLoss(alpha=grad_smooth_factor0)
    berhu_loss1 = BerHuLoss(alpha=grad_smooth_factor1)

    import numpy as np
    target1 = np.load('../dataset/oaisys-new/inv-depth-01-npy/00123Left.npy')
    target2 = np.load('../dataset/oaisys-new/inv-depth-01-npy/00055Left.npy')
    # target2 = np.load('/home/ch5225/Desktop/模拟数据/oaisys-new/depth_npy/00035Left.npy')
    # target = [target1, target2]
    target = torch.tensor([target1, target2])

    torch.manual_seed(2022)
    # predict = torch.randn_like(target)
    predict = target - torch.rand_like(target)

    mask = torch.zeros_like(target)
    mask[torch.where(target > 0)] = 1

    mask = torch.ones_like(target)

    loss1 = ssi_loss0(predict, target, mask=mask)

    loss2 = ssi_loss1(predict, target, mask=mask)

    loss3 = berhu_loss0(predict, target, mask=mask)

    loss4 = berhu_loss1(predict, target, mask=mask)

    param = berhu_loss1.awl.params[0].item()
