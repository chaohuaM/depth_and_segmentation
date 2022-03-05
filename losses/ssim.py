import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def type_trans(window, img):
    if img.is_cuda:
        window = window.cuda(img.get_device())
    return window.type_as(img)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    # print(mu1.shape,mu2.shape)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    mcs_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    # print(ssim_map.shape)
    if size_average:
        return ssim_map.mean(), mcs_map.mean()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=5, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        window = create_window(self.window_size, channel)
        window = type_trans(window, img1)
        ssim_map, mcs_map = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return ssim_map


class MultiScale_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(MultiScale_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        # self.channel = 3

    def forward(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))

        msssim = Variable(torch.Tensor(levels, ))
        mcs = Variable(torch.Tensor(levels, ))

        if torch.cuda.is_available():
            weight = weight.cuda()
            msssim = msssim.cuda()
            mcs = mcs.cuda()

        _, channel, _, _ = img1.size()
        window = create_window(self.window_size, channel)
        window = type_trans(window, img1)

        for i in range(levels):  # 5 levels
            ssim_map, mcs_map = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            # print(img1.shape)
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1  # refresh img
            img2 = filtered_im2

        return torch.prod((msssim[levels - 1] ** weight[levels - 1] * mcs[0:levels - 1] ** weight[0:levels - 1]))
        # return torch.prod((msssim[levels-1] * mcs[0:levels-1]))
        # torch.prod: Returns the product of all elements in the input tensor


# ########################  example ######################
if __name__ == '__main__':

    from torch import optim
    import cv2

    npImg1 = cv2.imread("/home/ch5225/chaohua/Rock-A/TW1-NaTeCam/train/20210603091742_00020_NaTeCamA-F-011.png")

    img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0  # 进行了归一化
    img2 = torch.rand(img1.size())
    print(img2.size())

    if torch.cuda.is_available():
        img1 = img1.cuda()
        print(img1.size())
        img2 = img2.cuda()
        print(img2.size())

    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=True)

    # *******************---  SSIM   ---*********************
    # Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)

    ssim_loss = SSIM()
    ssim_value = ssim_loss(img1, img2).data
    print("Initial ssim:", ssim_value)

    optimizer = optim.Adam([img2], lr=0.1)

    while ssim_value < 0.99:
        optimizer.zero_grad()
        ssim_out = -ssim_loss(img1, img2)
        ssim_value = - ssim_out.data
        print(ssim_value)
        out_img = img2.permute(0, 2, 3, 1).contiguous()[0].data.cpu().numpy()
        cv2.imshow('output', out_img)
        cv2.waitKey(10)
        ssim_out.backward()
        optimizer.step()

    # #######################  MS_SSIM ######################
    # ms_ssim_loss = MultiScale_SSIM()
    # optimizer = optim.Adam([img2], lr=0.01)
    #
    # ms_ssim_value = ms_ssim_loss(img1, img2).data
    # print("Initial ssim:", ms_ssim_value)

    # while ms_ssim_value < 0.99:
    #     optimizer.zero_grad()
    #     ms_ssim_out = -ms_ssim_loss(img1, img2)
    #     ms_ssim_value = - ms_ssim_out.data
    #     out_img = img2.permute(0, 2, 3, 1).contiguous()[0].data.cpu().numpy()
    #     cv2.imshow('output', out_img)
    #     cv2.waitKey(10)
    #     print(ms_ssim_value)
    #     ms_ssim_out.backward()
    #     optimizer.step()
