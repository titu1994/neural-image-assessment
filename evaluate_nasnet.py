import argparse
import csv

import numpy as np
from path import Path
from tqdm import tqdm

from config import weights_file
from nasnet_model import *
from utils.image_utils import preprocess_for_evaluation
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

# load model
nima_model = NimaModel()
model = nima_model.model
model.load_weights(weights_file)

# calculate scores
scored_images = []
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

# write results to csv file
with open('results.csv', 'w', encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=';', lineterminator='\n')
    csvwriter.writerow(['filename', 'mean', 'std'])
    for mean, std, img_path in scored_images:
        print("{:.3f} +- ({:.3f})  {}".format(mean, std, img_path))
        csvwriter.writerow([img_path, mean, std])
