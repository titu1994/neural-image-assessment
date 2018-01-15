import argparse
import csv

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Model
from path import Path
from tqdm import tqdm

from utils.image_utils import preprocess_for_evaluation
from utils.nasnet import NASNetMobile
from utils.score_utils import mean_score, std_score

parser = argparse.ArgumentParser(
    description='Evaluate NIMA(NASNet mobile)')
parser.add_argument('--dir', type=str, default=None,
                    help='Pass a directory to evaluate the images in it')

parser.add_argument('--img', type=str, default=[None], nargs='+',
                    help='Pass one or more image paths to evaluate them')

args = parser.parse_args()


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
    raise RuntimeError(
        'Either --dir or --img arguments must be passed as argument')

scored_images = []
with tf.device('/GPU:0'):
    base_model = NASNetMobile((224, 224, 3), include_top=False, pooling='avg',
                              weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)
    model = Model(base_model.input, x)
    model.load_weights('weights/nasnet_weights.h5')
    for img_path in tqdm(imgs):
        try:
            x = preprocess_for_evaluation(img_path)
        except OSError as e:
            print("Couldn't process {}".format(img_path))
            print(e)
            continue
        x = np.expand_dims(x, axis=0)
        scores = model.predict(x, batch_size=1, verbose=0)[0]

        mean = mean_score(scores)
        std = std_score(scores)

        scored_images.append((mean, std, img_path))
    scored_images = sorted(scored_images, reverse=True)
    with open('results.csv', 'w', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', lineterminator='\n')
        csvwriter.writerow(['filename', 'mean', 'std'])
        for mean, std, img_path in scored_images:
            print("{:.3f} +- ({:.3f})  {}".format(mean, std, img_path))
            csvwriter.writerow([img_path, mean, std])
