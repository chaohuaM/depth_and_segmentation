# @Author  : ch
# @Time    : 2022/5/20 下午12:37
# @File    : get_tb_figures.py
import os.path

import numpy as np
from PIL import Image
import imageio
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


def fig2data(fig):
    """
    # @Author : panjq
    # @E-mail : pan_jinquan@163.com
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image


event_filename = '../test-logs/unet_dual_decoder_with_sa/2022_05_20_01_43_19/events.out.tfevents.1652982203.ch5225.5274.0'
out_dir = './dsa-map/'
image_tags = ['1sa-map-0', '1sa-map-1', '1sa-map-2', '1sa-map-3', '1sa-map-4']

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


ims = []

for image_tag in image_tags:
    count = 0
    for event in summary_iterator(event_filename):
        for v in event.summary.value:
            if v.tag == image_tag:
                a = tf.image.decode_image(v.image.encoded_image_string).numpy()[:, :, 0]
                a = a / 255
                fig = plt.figure()
                plt.imshow(a, cmap='jet')
                plt.title('epoch: ' + str(count))
                plt.colorbar()
                plt.savefig(os.path.join(out_dir, image_tag+"-"+str(count) + ".png"))
                plt.close()
                im = fig2data(fig)
                ims.append(im)
                count += 1

    imageio.mimsave(os.path.join(out_dir, image_tag + ".gif"), ims, fps=5)
    print(image_tag + '.gif saved!')
