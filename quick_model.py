from keras.layers import Conv2D, Dense, Dropout, Input, MaxPooling2D, Flatten
from keras.models import Model


class NimaModel(object):
    def __init__(self):
        inputs = Input(shape=(224, 224, 3))
        x = Conv2D(16, kernel_size=(3, 3),
                   activation='relu')(inputs)
        x = Conv2D(8, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        self.base_model = Model(inputs, x)
        x = Dropout(0.75)(x)
        x = Dense(10, activation='softmax', name='toplayer')(x)
        self.model = Model(inputs, x)
        self.model.summary()
