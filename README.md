# NIMA: Neural Image Assessment
Implementation of [NIMA: Neural Image Assessment](https://arxiv.org/abs/1709.05424) in Keras + Tensorflow with weights for MobileNet model trained on AVA dataset.

NIMA assigns a Mean + Standard Deviation score to images, and can be used as a tool to automatically inspect quality of images or as a loss function to further improve the quality of generated images.

# Example
<img src="https://github.com/titu1994/neural-image-assessment/blob/master/images/NIMA.jpg?raw=true" height=100% width=100%>

<img src="https://github.com/titu1994/neural-image-assessment/blob/master/images/NIMA2.jpg?raw=true" height=100% width=100%>

# Requirements
- Keras
- Tensorflow (CPU to evaluate, GPU to train)
- skimage
