import torch
import numpy as np
import torch.nn as nn


def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)
    # grad y
    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x


def imgrad_yx(img):
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)


def GRAD_LOSS(z, z_fake):
    grad_real, grad_fake = imgrad_yx(z), imgrad_yx(z_fake)

    # L1 norm
    return torch.mean(torch.abs(grad_real - grad_fake))


def _mask_input(input, mask=None):
    if mask is not None:
        input = input * mask
        count = torch.sum(mask).data[0]
    else:
        count = np.prod(input.size(), dtype=np.float32).item()
    return input, count


def BerHu_Loss(z, z_fake, mask=None):
    x = z_fake - z
    abs_x = torch.abs(x)
    c = torch.max(abs_x).item() / 5.0
    leq = (abs_x <= c).float()
    l2_losses = (x ** 2 + c ** 2) / (2 * c)
    losses = leq * abs_x + (1 - leq) * l2_losses
    losses, count = _mask_input(losses, mask)
    return torch.sum(losses) / count









