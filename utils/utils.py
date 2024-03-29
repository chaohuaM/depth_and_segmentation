import numpy as np
from PIL import Image
import cv2


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

    # ---------------------------------------------------#


# 读取深度图像，深度图格式为exr
def load_exr(image_path):
    """
    读取深度图像，深度图格式为exr
    :param image_path:  str, the path of depth image, end with '.exr'
    :return:  opencv ndarray object, type = float32
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = image[:, :, 0]
    image[image == 10000000000.00000] = 0.0

    return image


#   对输入图像进行resize
# ---------------------------------------------------#
def resize_and_centered(image, size, reverse=False):
    """
    将图像进行按比例缩放，若小于size，则复制在图像中心
    :param reverse: 使用时是否为变换为原图尺寸
    :param image: ndarray 原图
    :param size: [int, int] [height, width] [rows, cols]
    :return: resize image and place to the center, ndarray
    """

    ih = image.shape[0]
    iw = image.shape[1]
    # 若等于原图大小，直接返回image
    if [ih, iw] == size:
        return image

    if reverse:
        return restore_image_size(image, size)

    h, w = size
    if len(image.shape) == 3:
        ic = image.shape[2]
        new_image = np.zeros([h, w, ic], dtype=image.dtype)
    else:
        new_image = np.zeros([h, w], dtype=image.dtype)

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    h0 = (h - nh) // 2
    w0 = (w - nw) // 2

    image = cv2.resize(image, (nw, nh))

    new_image = paste_image(image, new_image, h0, w0)

    return new_image


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image /= 255.0
    return image


def paste_image(front, back, h0, w0):
    """
    粘贴图片，超出范围也可以
    :param front: 前景
    :param back: 背景
    :param h0: 粘贴点左上角坐标，对应行
    :param w0: 粘贴点左上角坐标，对应列
    :return:
    """
    front = Image.fromarray(front)
    back = Image.fromarray(back)

    back.paste(front, (w0, h0))
    back = np.array(back)

    return back


def normalization(data):
    mi = np.min(data)
    ma = np.max(data)
    _range = ma - mi
    return (data - mi) / _range


def restore_image_size(image, original_size):
    h = image.shape[0]
    w = image.shape[1]
    # 若等于原图大小，直接返回image
    if [h, w] == original_size:
        return image

    ih, iw = original_size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    h0 = (h - nh) // 2
    w0 = (w - nw) // 2

    new_image = image[h0:h0 + nh, w0:w0 + nw]
    new_image = cv2.resize(new_image, (iw, ih), interpolation=cv2.INTER_LINEAR)

    return new_image


# 将pytorch模型权重加载到模型，并导出onnx模型
def pth2onnx(model, input_shape, onnx_path):
    """
    export onnx model from pytorch CNN model with weights loaded
    :param model: pytorch CNN model
    :param input_shape: model input tensor shape [N,C,H,W]
    :param onnx_path:  the exported onnx model path
    :return: None
    """
    import torch.onnx

    model = model.eval()
    in_tensor = torch.randn(input_shape, requires_grad=True).to('cuda' if torch.cuda.is_available() else 'cpu')

    torch.onnx.export(model,  # model being run
                      in_tensor,  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
