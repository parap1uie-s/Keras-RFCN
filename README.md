# Keras-RFCN
RFCN implement based on Keras&amp;Tensorflow

This is an implementation of [Region-based Fully Convolutional Networks](https://arxiv.org/pdf/1605.06409v2.pdf) on Python 3, Keras, and TensorFlow. The model generates bounding boxes for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a [ResNet50](https://arxiv.org/abs/1512.03385) or ResNet101 backbone.

The repository includes:

* Source code of RFCN built on FPN and ResNet50/101.
* Training code for [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) with 46 clothes classes.
* Pre-trained weights for [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) - Uploading
* Example of training on your own dataset&nbsp;-&nbsp;see Fashion_Train.py and Fashion_Test.py


# Getting Started

Thanks to the [Mask-RCNN implement by matterport](https://github.com/matterport/Mask_RCNN), we have the great framework so that we don't have the needs to generate bounding box and implement the Non-Maximum-Suppression algorithm.

If you are already fimilar with matterport's framework, this repository is easy to understand and use. What I have done is remove the mask head in Mask-RCNN, and implement a position sensitive ROI pooling layer and a VOTE layer. For more details, please read the [paper](https://arxiv.org/pdf/1605.06409v2.pdf).

![position sensitive ROI](ReadmeImages/1.png)

As you can see in **Fashion_Train.py**, 