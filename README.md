# NIMA: Neural Image Assessment
Implementation of [NIMA: Neural Image Assessment](https://arxiv.org/abs/1709.05424) in Keras + Tensorflow with weights for MobileNet model trained on AVA dataset.

NIMA assigns a Mean + Standard Deviation score to images, and can be used as a tool to automatically inspect quality of images or as a loss function to further improve the quality of generated images.

Contains weights trained on the AVA dataset for the following models:
- NASNet Mobile (0.067 EMD on valset thanks to [@tfriedel](https://github.com/tfriedel) !, 0.0848 EMD with just pre-training)
- Inception ResNet v2 (~ 0.07 EMD on valset, thanks to [@tfriedel](https://github.com/tfriedel) !)
- MobileNet (0.0804 EMD on valset)

# Usage
## Evaluation
There are `evaluate_*.py` scripts which can be used to evaluate an image using a specific model. The weights for the specific model must be downloaded from the [Releases Tab](https://github.com/titu1994/neural-image-assessment/releases) and placed in the weights directory.

Supports either passing a directory using `--dir` or a set of full paths of specific images using `--img` (seperate multiple image paths using spaces between them)

There's also a script sort_lightroom_collection.py, which will sort a Adobe Lightroom collection using the calculated scores. For an example see the screenshots.


### Arguments:
```
--dir    : Pass the relative/full path of a directory containing a set of images. Only png, jpg and jpeg images will be scored.
--img    : Pass one or more relative/full paths of images to score them. Can support all image types supported by PIL.
```

## Training
The AVA dataset is required for training these models. I used about 240.000 images to train and the last 10.000 images to evaluate (this is not the same format as in the paper).
You can download it as a torrent from here:
http://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460

First, ensure that the dataset is clean - no corrupted JPG files etc by using the `check_dataset.py` script in the utils folder. If such currupted images exist, it will drastically slow down training since the Tensorflow Dataset buffers will constantly flush and reload on each occurance of a corrupted image.
Then just run train_nasnet_mobile.py.


# Example
## best ranked images
<img src="https://github.com/tfriedel/neural-image-assessment/blob/master/images/top_images.jpg?raw=true" height=100% width=100%>

## worst ranked images
<img src="https://github.com/tfriedel/neural-image-assessment/blob/master/images/bottom_images.jpg?raw=true" height=100% width=100%>

# Requirements
- Keras
- Tensorflow (CPU to evaluate, GPU to train)
- path.py
- tqdm
- pillow

