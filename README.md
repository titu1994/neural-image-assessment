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

Supports either passing a directory using `-dir` or a set of full paths of specific images using `-img` (seperate multiple image paths using spaces between them)

Supports passing an argument `-resize "true/false"` to resize each image to (224x224) or not before passing for NIMA scoring. 
**Note** : NASNet models do not support this argument, all images **must be resized prior to scoring !**

### Arguments: 
```
-dir    : Pass the relative/full path of a directory containing a set of images. Only png, jpg and jpeg images will be scored.
-img    : Pass one or more relative/full paths of images to score them. Can support all image types supported by PIL.
-resize : Pass "true" or "false" as values. Resize an image prior to scoring it. Not supported on NASNet models.
```

## Training
The AVA dataset is required for training these models. I used 250,000 images to train and the last 5000 images to evaluate (this is not the same format as in the paper).

First, ensure that the dataset is clean - no currupted JPG files etc by using the `check_dataset.py` script in the utils folder. If such currupted images exist, it will drastically slow down training since the Tensorflow Dataset buffers will constantly flush and reload on each occurance of a currupted image.

Then, there are two ways of training these models.
### Direct-Training
In direct training, you have to ensure that the model can be loaded, trained, evaluated and then saved all on a single GPU. If this cannot be done (because the model is too large), refer to the Pretraining section.

Use the `train_*.py` scripts for direct training. Note, if you want to train other models, copy-paste a train script and only edit the `base_model` creation part, everythin else should likely be the same.

### Pre-Training
If the model is too large to train directly, training can still be done in a roundabout way (as long as you are able to do inference with a batch of images with the model).

**Note** : One obvious drawback of such a method is that it wont have the performance of the full model without further finetuning. 

This is a 3 step process:

1)  **Extract features from the model**: Use the `extract_*_features.py` script to extract the features from the large model. In this step, you can change the batch_size to be small enough to not overload your GPU memory, and save all the features to 2 TFRecord objects.

2) **Pre-Train the model**: Once the features have been extracted, you can simply train a small feed forward network on those features directly. Since the feed forward network will likely easily fit onto memory, you can use large batch sizes to quickly train the network.

3) **Fine-Tune the model**: This step is optional, only for those who have sufficient memory to load both the large model and the feed forward classifier at the same time. Use the `train_nasnet_mobile.py` as reference as to how to load both the large model and the weights of the feed forward network into this large model and then train fully for several epochs at a lower learning rate.

# Example
<img src="https://github.com/titu1994/neural-image-assessment/blob/master/images/NIMA.jpg?raw=true" height=100% width=100%>

<img src="https://github.com/titu1994/neural-image-assessment/blob/master/images/NIMA2.jpg?raw=true" height=100% width=100%>

# Requirements
- Keras
- Tensorflow (CPU to evaluate, GPU to train)
- Numpy
- Path.py
- PIL
