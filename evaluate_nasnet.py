import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from utils.nasnet import NASNetMobile, preprocess_input
from utils.score_utils import mean_score, std_score

with tf.device('/CPU:0'):
    base_model = NASNetMobile((224, 224, 3), include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('weights/nasnet_weights.h5', by_name=True)

    img_path = 'images/art1.jpg'
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    scores = model.predict(x, batch_size=1, verbose=1)[0]

    mean = mean_score(scores)
    std = std_score(scores)

    print("Evaluating : ", img_path)
    print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))


