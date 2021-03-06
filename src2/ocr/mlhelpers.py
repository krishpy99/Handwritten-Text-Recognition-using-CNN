import numpy as np
import math
import matplotlib.pyplot as plt
import csv

class TrainingPlot:
    train_loss = []
    train_acc = []
    valid_acc = []
    test_iter = 0
    loss_iter = 0
    interval = 0
    ax1 = None
    ax2 = None
    fig = None
    def __init__(self, steps, test_itr, loss_itr):
        self.test_iter = test_itr
        self.loss_iter = loss_itr
        self.interval = steps
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()
        self.ax1.set_autoscaley_on(True)
        plt.ion()
        self._update_plot()
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Train Loss')
        self.ax2.set_ylabel('Valid. Accuracy')
        self.ax1.set_ylim([0,10])

    def _update_plot(self):
        self.fig.canvas.draw()

    def update_loss(self, loss_train, index):
        self.trainLoss.append(loss_train)
        if len(self.train_loss) == 1:
            self.ax1.set_ylim([0, min(10, math.ceil(loss_train))])
        self.ax1.plot(self.lossInterval * np.arange(len(self.train_loss)),self.train_loss, 'b', linewidth=1.0)
        self.updatePlot()

    def update_acc(self, acc_val, acc_train, index):
        self.validAcc.append(acc_val)
        self.trainAcc.append(acc_train)
        self.ax2.plot(self.test_iter * np.arange(len(self.valid_acc)), self.valid_acc, 'r', linewidth=1.0)
        self.ax2.plot(self.test_iter * np.arange(len(self.train_acc)), self.train_acc, 'g',linewidth=1.0)
        self.ax2.set_title('Valid. Accuracy: {:.4f}'.format(self.valid_acc[-1]))
        self.updatePlot()

class DataSet:
    images = None
    labels = None
    length = 0
    index = 0
    def __init__(self, img, lbl):
        self.images = img
        self.labels = lbl
        self.length = len(img)
        self.index = 0
    def next_batch(self, batch_size):
        start = self.index
        self.index += batch_size
        if self.index > self.length:
            perm = np.arange(self.length)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            start = 0
            self.index = batch_size
        end = self.index
        return self.images[start:end], self.labels[start:end]
