# Keras-RFCN
RFCN implement based on Keras&amp;Tensorflow

This is an implementation of [Region-based Fully Convolutional Networks](https://arxiv.org/pdf/1605.06409v2.pdf) on Python 3, Keras, and TensorFlow. The model generates bounding boxes for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a [ResNet50](https://arxiv.org/abs/1512.03385) or ResNet101 backbone.

The repository includes:

	* Source code of RFCN built on FPN and ResNet50/101.
	* Training code for [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
	* Pre-trained weights for [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
	* Jupyter notebooks to visualize the detection pipeline at every step
	* ParallelModel class for multi-GPU training
	* Evaluation on MS COCO metrics (AP)
	* Example of training on your own dataset