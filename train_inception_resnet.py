import os

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import backend as K

from utils.data_loader import train_generator, val_generator

'''
Below is a modification to the TensorBoard callback to perform 
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''
class TensorBoardBatch(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

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

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch * self.batch_size)

        self.writer.flush()

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

image_size = 224

base_model = InceptionResNetV2(input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False

x = Dropout(0.75)(base_model.output)
x = Dense(10, activation='softmax')(x)

model = Model(base_model.input, x)
model.summary()
optimizer = Adam(lr=1e-3)
model.compile(optimizer, loss=earth_mover_loss)

# load weights from trained model if it exists
if os.path.exists('weights/inception_resnet_weights.h5'):
    model.load_weights('weights/inception_resnet_weights.h5')

# load pre-trained NIMA(Inception ResNet V2) classifier weights
# if os.path.exists('weights/inception_resnet_pretrained_weights.h5'):
#     model.load_weights('weights/inception_resnet_pretrained_weights.h5', by_name=True)

checkpoint = ModelCheckpoint('weights/inception_resnet_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,
                             mode='min')
tensorboard = TensorBoardBatch()
callbacks = [checkpoint, tensorboard]

batchsize = 100
epochs = 20

model.fit_generator(train_generator(batchsize=batchsize),
                    steps_per_epoch=(250000. // batchsize),
                    epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_generator(batchsize=batchsize),
                    validation_steps=(5000. // batchsize))