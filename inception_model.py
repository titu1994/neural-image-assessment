from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.applications.inception_v3 import InceptionV3
from config import IMAGE_SIZE

class InceptionModel(object):
    def __init__(self):
        self.base_model = InceptionV3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                  include_top=False, pooling='avg',
                                  weights='imagenet', weight_decay=0, dropout=0)
        x = Dropout(0.75)(self.base_model.output)
        x = Dense(10, activation='softmax', name='toplayer')(x)

        self.model = Model(self.base_model.input, x)