from keras.callbacks import Callback
import numpy as np
from keras import backend as K

class SGDRScheduler(Callback):
    def __init__(self,
                 epochsize,
                 batchsize,
                 start_epoch=0,
                 t_e=10,
                 t_0=np.pi / 2.,
                 mult_factor=2,
                 lr_fac=0.1,
                 lr_reduction_epochs=[]):
        super(SGDRScheduler, self).__init__()
        self.epoch = -1
        self.batch = 0
        self.tt = 0
        self.te_next = t_e
        self.epochsize = epochsize
        self.batchsize = batchsize
        self.start_epoch = start_epoch
        self.t_e = t_e
        self.t_0 = t_0
        self.mult_factor = mult_factor
        self.lr_fac = lr_fac
        self.lr_reduction_epochs = lr_reduction_epochs
        self.lr_log = []

    def on_train_begin(self, logs={}):
        self.lr = K.get_value(self.model.optimizer.lr)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1

    def on_batch_end(self, batch, logs={}):
        self.lr_log.append(K.get_value(self.model.optimizer.lr))
        self.batch += 1

        if (self.epoch + 1 >= self.start_epoch):
            dt = np.pi / self.t_e
            self.tt += dt / (self.epochsize / self.batchsize)
            if self.tt >= np.pi:
                self.tt -= np.pi
            cur_t = self.t_0 + self.tt
            lr = self.lr * (1. + np.sin(cur_t)) / 2.
            K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch + 1 == self.te_next:
            self.tt = 0
            self.t_e *= self.mult_factor
            self.te_next += self.t_e

        if (self.epoch + 1) in self.lr_reduction_epochs:
            self.lr *= self.lr_fac