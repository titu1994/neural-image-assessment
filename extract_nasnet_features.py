import numpy as np

import tensorflow as tf
from keras import backend as K
from nasnet import NASNetMobile

from data_loader import train_generator, val_generator

sess = tf.Session()
K.set_session(sess)

image_size = 224

def _float32_feature_list(floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=floats))

model = NASNetMobile((image_size, image_size, 3), include_top=False, pooling='avg')
model.summary()

# ''' TRAIN SET '''
nb_samples = 250000 * 2
batchsize = 200

with sess.as_default():
    generator = train_generator(batchsize, shuffle=False)
    writer = tf.python_io.TFRecordWriter('weights/nasnet_train.tfrecord')

count = 0
for _ in range(nb_samples // batchsize):
    x_batch, y_batch = next(generator)

    with sess.as_default():
        x_batch = model.predict(x_batch, batchsize, verbose=1)

    for i, (x, y) in enumerate(zip(x_batch, y_batch)):
        examples = {
            'features': _float32_feature_list(x.flatten()),
            'scores': _float32_feature_list(y.flatten()),
        }
        features = tf.train.Features(feature=examples)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

    count += batchsize

    print("Finished %0.2f percentage storing dataset" % (count  * 100 / float(nb_samples)))

writer.close()

''' TRAIN SET '''
nb_samples = 5000
batchsize = 200

with sess.as_default():
    generator = val_generator(batchsize)
    writer = tf.python_io.TFRecordWriter('weights/nasnet_val.tfrecord')

count = 0
for _ in range(nb_samples // batchsize):
    x_batch, y_batch = next(generator)

    with sess.as_default():
        x_batch = model.predict(x_batch, batchsize, verbose=1)

    for i, (x, y) in enumerate(zip(x_batch, y_batch)):
        examples = {
            'features': _float32_feature_list(x.flatten()),
            'scores': _float32_feature_list(y.flatten()),
        }
        features = tf.train.Features(feature=examples)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

    count += batchsize

    print("Finished %0.2f percentage storing dataset" % (count  * 100 / float(nb_samples)))

writer.close()