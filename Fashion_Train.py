"""
Keras RFCN
Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by parap1uie-s@github.com
"""

'''
This is a demo to TRAIN a RFCN model with DeepFashion Dataset
http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
'''

from KerasRFCN.Model.Model import RFCN_Model
from KerasRFCN.Config import Config
from KerasRFCN.Utils import Dataset
import os
import pickle
import numpy as np
from PIL import Image

############################################################
#  Config
############################################################

class RFCNNConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Fashion"

    # Backbone model
    # choose one from ['resnet50', 'resnet101', 'resnet50_dilated', 'resnet101_dilated']
    BACKBONE = "resnet101"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    C = 1 + 46  # background + 2 tags
    NUM_CLASSES = C
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 768

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    # Use same strides on stage 4-6 if use dilated resnet of DetNet
    # Like BACKBONE_STRIDES = [4, 8, 16, 16, 16]
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 200

    RPN_NMS_THRESHOLD = 0.6
    POOL_SIZE = 7

############################################################
#  Dataset
############################################################

class FashionDataset(Dataset):
    # count - int, images in the dataset
    def initDB(self, count, start = 0):
        self.start = start

        all_images, classes_count, class_mapping = pickle.load(open("data.pk", "rb"))
        self.classes = {}
        # Add classes
        for k,c in class_mapping.items():
            self.add_class("Fashion",c,k)
            self.classes[c] = k

        for k, item in enumerate(all_images[start:count+start]):
            self.add_image(source="Fashion",image_id=k, path=item['filepath'], width=item['width'], height=item['height'], bboxes=item['bboxes'])

        self.rootpath = '/content/'

    # read image from file and get the 
    def load_image(self, image_id):
        info = self.image_info[image_id]
        # tempImg = image.img_to_array( image.load_img(info['path']) )
        tempImg = np.array(Image.open( os.path.join(self.rootpath, info['path']) ))
        return tempImg

    def get_keys(self, d, value):
        return [k for k,v in d.items() if v == value]

    def load_bbox(self, image_id):
        info = self.image_info[image_id]
        bboxes = []
        labels = []
        for item in info['bboxes']:
            bboxes.append((item['y1'], item['x1'], item['y2'], item['x2']))
            label_key = self.get_keys(self.classes, item['class'])
            if len(label_key) == 0:
                continue
            labels.extend( label_key )
        return np.array(bboxes), np.array(labels)

if __name__ == '__main__':
    ROOT_DIR = os.getcwd()

    config = RFCNNConfig()
    dataset_train = FashionDataset()
    dataset_train.initDB(100000)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FashionDataset()
    dataset_val.initDB(5000, start=100000)
    dataset_val.prepare()

    model = RFCN_Model(mode="training", config=config, model_dir=os.path.join(ROOT_DIR, "logs") )

    # This is a hack, bacause the pre-train weights are not fit with dilated ResNet
    model.keras_model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True, skip_mismatch=True)

    try:
        model_path = model.find_last()[1]
        if model_path is not None:
            model.load_weights(model_path, by_name=True)
    except Exception as e:
        print(e)
        print("No checkpoint founded")
        
    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=80,
                layers='all')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=240,
                layers='all')