from keras.callbacks import TensorBoard
from keras import backend as K

class TensorBoardBatch(TensorBoard):
    """ modification to the TensorBoard callback to perform
        batchwise writing to the tensorboard, instead of only at the end
        of the batch.
    """
    def __init__(self, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args, **kwargs)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, batch)

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch * self.batch_size)

        self.writer.flush()
