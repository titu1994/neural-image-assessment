import numpy as np
import argparse
from path import Path

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from utils.nasnet import NASNetMobile, preprocess_input
from utils.score_utils import mean_score, std_score

parser = argparse.ArgumentParser(description='Evaluate NIMA(Inception ResNet v2)')
parser.add_argument('-dir', type=str, default=None,
                    help='Pass a directory to evaluate the images in it')

parser.add_argument('-img', type=str, default=[None], nargs='+',
                    help='Pass one or more image paths to evaluate them')

args = parser.parse_args()
target_size = (224, 224)  # NASNet requires strict size set to 224x224

# give priority to directory
if args.dir is not None:
    print("Loading images from directory : ", args.dir)
    imgs = Path(args.dir).files('*.png')
    imgs += Path(args.dir).files('*.jpg')
    imgs += Path(args.dir).files('*.jpeg')

elif args.img[0] is not None:
    print("Loading images from path(s) : ", args.img)
    imgs = args.img

else:
    raise RuntimeError('Either -dir or -img arguments must be passed as argument')

with tf.device('/CPU:0'):
    base_model = NASNetMobile((224, 224, 3), include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('weights/nasnet_weights.h5')

    for img_path in imgs:
        img = load_img(img_path, target_size=target_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)

        scores = model.predict(x, batch_size=1, verbose=0)[0]

        mean = mean_score(scores)
        std = std_score(scores)

        print("Evaluating : ", img_path)
        print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
        print()


