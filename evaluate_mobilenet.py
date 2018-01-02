import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf

from utils import mean_score, std_score

with tf.device('/CPU:0'):
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('weights/mobilenet_weights.h5')

    img_path = 'images/img.png'
    img = load_img(img_path)
    x = img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    scores = model.predict(x, batch_size=1, verbose=1)[0]

    mean = mean_score(scores)
    std = std_score(scores)

    print("Evaluating : ", img_path)
    print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))


