"""
Keras RFCN
Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by parap1uie-s@github.com
"""

'''
This is a demo to Eval a RFCN model with DeepFashion Dataset
http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
'''

from KerasRFCN.Model.Model import RFCN_Model
from KerasRFCN.Config import Config
import KerasRFCN.Utils 
import os
from keras.preprocessing import image
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RFCNNConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Fashion"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    C = 1 + 46  # background + 2 tags
    NUM_CLASSES = C
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 300
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    RPN_NMS_THRESHOLD = 0.7

    DETECTION_MIN_CONFIDENCE = 0.4
    POOL_SIZE = 7


def Test(model, loadpath, savepath):
    assert not loadpath == savepath, "loadpath should'n same with savepath"

    model_path = model.find_last()[1]
    # Load trained weights (fill in path to trained weights here)
    
    model.load_weights(model_path, by_name=True)
    print("Loading weights from ", model_path)

    if os.path.isdir(loadpath):
        for idx, imgname in enumerate(os.listdir(loadpath)):
            if not imgname.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(imgname)
            imageoriChannel = np.array(plt.imread( os.path.join(loadpath, imgname) )) / 255.0
            img = image.img_to_array( image.load_img(os.path.join(loadpath, imgname)) )
            TestSinglePic(img, imageoriChannel, model, savepath=savepath, imgname=imgname)
            
    elif os.path.isfile(loadpath):
        if not loadpath.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            print("not image file!")
            return
        print(loadpath)
        imageoriChannel = np.array(plt.imread( loadpath )) / 255.0
        img = image.img_to_array( image.load_img(loadpath) )
        (filename,extension) = os.path.splitext(loadpath)
        TestSinglePic(img, imageoriChannel, model, savepath=savepath, imgname=filename)
    
def TestSinglePic(image, image_ori, model, savepath, imgname):
    r = model.detect([image], verbose=1)[0]
    print(r)
    def get_ax(rows=1, cols=1, size=8):
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax

    ax = get_ax(1)

    assert not savepath == "", "empty save path"
    assert not imgname == "", "empty image file name"

    for box in r['rois']:
        y1, x1, y2, x2 = box
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor="red", facecolor='none')
        ax.add_patch(p)
    ax.imshow(image_ori)

    plt.savefig(os.path.join(savepath, imgname),bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    parser = argparse.ArgumentParser()

    parser.add_argument('--loadpath', required=False,
                default="images/",
                metavar="evaluate images loadpath",
                help="evaluate images loadpath")
    parser.add_argument('--savepath', required=False,
            default="result/",
            metavar="evaluate images savepath",
            help="evaluate images savepath")

    config = RFCNNConfig()
    args = parser.parse_args()

    model = RFCN_Model(mode="inference", config=config,
                      model_dir=os.path.join(ROOT_DIR, "logs") )

    Test(model, args.loadpath, args.savepath)