from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from config import IMAGE_SIZE

class NimaModel(object):
    def __init__(self):
        self.base_model = InceptionResNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                  include_top=False, pooling='avg',
                                  weights='imagenet')
        x = Dropout(0.75)(self.base_model.output)
        x = Dense(10, activation='softmax', name='toplayer')(x)

        self.model = Model(self.base_model.input, x)