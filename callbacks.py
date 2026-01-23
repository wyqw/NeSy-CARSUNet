import os
import matplotlib
import datetime
from matplotlib import pyplot as plt
import scipy.signal
matplotlib.use('Agg')

class LossHistory():
    def __init__(self):
        self.log_dir = "./train_logs"
        self.losses = []
        self.iou = []
        self.dice = []
        self.id = datetime.datetime.now().strftime("%m%d-%H%M")

    def append_loss(self, loss=None, iou=None, dice=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.losses.append(loss)
        self.iou.append(iou)
        self.dice.append(dice)

        loss_dir = self.log_dir + '/loss'
        iou_dir = self.log_dir + '/iou'
        dice_dir = self.log_dir + '/dice'

        with open(os.path.join(loss_dir, "epoch_loss" + self.id + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(iou_dir, "epoch_iou" + self.id + ".txt"), 'a') as f:
            f.write(str(iou))
            f.write("\n")
        with open(os.path.join(dice_dir, "epoch_dice" + self.id + ".txt"), 'a') as f:
            f.write(str(dice))
            f.write("\n")

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        loss_dir = self.log_dir + '/loss'
        iou_dir = self.log_dir + '/iou'
        dice_dir = self.log_dir + '/dice'
        plt.figure()
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(loss_dir, "epoch_loss" + self.id + ".png"))

        plt.cla()
        plt.close("all")

        iters = range(len(self.iou))
        plt.figure()
        plt.plot(iters, self.iou, 'red', linewidth = 2, label='val iou')
        try:
            if len(self.iou) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.iou, num, 3), 'coral', linestyle='--', linewidth=2,
                     label='smooth val iou')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('iou')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(iou_dir, "epoch_iou" + self.id + ".png"))

        plt.cla()
        plt.close("all")

        iters = range(len(self.dice))

        plt.figure()

        try:
            if len(self.dice) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.dice, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth val dice')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('dice')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(dice_dir, "epoch_dice" + self.id + ".png"))

        plt.cla()
        plt.close("all")
