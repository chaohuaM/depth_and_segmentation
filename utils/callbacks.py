import os

import scipy.signal
from matplotlib import pyplot as plt


class LossHistory:
    def __init__(self, log_dir, val_loss_flag=True):
        self.time_str = log_dir.split('/')[-1]
        self.save_path = log_dir
        self.model_save_path = log_dir + '/checkpoints/'
        self.val_loss_flag = val_loss_flag

        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def log_dict(self, dict, stage):
        """

        :param dict:
        :return:
        """
        for name, value in dict.items():
            with open(os.path.join(self.save_path, stage+'-'+name+".txt"), 'a') as f:
                f.write(str(value))
                f.write("\n")

    def append_loss(self, loss, val_loss=0):
        self.losses.append(loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")

        if self.val_loss_flag:
            self.val_loss.append(val_loss)
            with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")
        self.loss_plot()

    def append_metric(self, metric):
        # TODO 添加记录metric的方式
        pass

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()

        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        try:
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, 5 if len(self.losses) < 25 else 15, 3), 'green',
                     linestyle='--', linewidth=2, label='smooth train loss')
        except:
            pass

        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
            try:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, 5 if len(self.losses) < 25 else 15, 3),
                         '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
            except:
                pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")
