"""
Keras RFCN
Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by parap1uie-s@github.com
"""

'''
This is Main class of RFCN Model
Contain the model's framework and call the backbone
'''

from KerasRFCN.Model.ResNet import ResNet
from KerasRFCN.Model.ResNet_dilated import ResNet_dilated
from KerasRFCN.Model.BaseModel import BaseModel
import KerasRFCN.Utils
import KerasRFCN.Losses

import keras.layers as KL
import keras.engine as KE
import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
import keras.models as KM

class RFCN_Model(BaseModel):
    """docstring for RFCN_Model"""
    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = ""
        self.build(mode=mode, config=config)

    def build(self, mode, config):
        # Inputs
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            h, w = K.shape(input_image)[1], K.shape(input_image)[2]
            image_scale = K.cast(K.stack([h, w, h, w], axis=0), tf.float32)
            gt_boxes = KL.Lambda(lambda x: x / image_scale)(input_gt_boxes)

        P2, P3, P4, P5, P6 = ResNet_dilated(input_image, architecture='resnet50').output_layers

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        ### RPN ###
        rpn = self.build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), 256)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        self.anchors = KerasRFCN.Utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                              config.RPN_ANCHOR_RATIOS,
                                              config.BACKBONE_SHAPES,
                                              config.BACKBONE_STRIDES,
                                              config.RPN_ANCHOR_STRIDE)
        # window size K and total classed num C
        # Example: For coco, C = 80+1
        scoreMapSize = config.K * config.K
        ScoreMaps_classify = []
        for feature_map_count, feature_map in enumerate(mrcnn_feature_maps):
            # [W * H * class_num] * k^2
            ScoreMap = KL.Conv2D(config.C * scoreMapSize, kernel_size=(1,1), name="score_map_class_{}".format(feature_map_count), padding='valid')(feature_map)
            ScoreMaps_classify.append(ScoreMap)

        ScoreMaps_regr = []
        for feature_map_count, feature_map in enumerate(mrcnn_feature_maps):
            # [W * H * 4] * k^2 ==> 4 = (x,y,w,h)
            ScoreMap = KL.Conv2D(4 * scoreMapSize, kernel_size=(1,1), name="score_map_regr_{}".format(feature_map_count), padding='valid')(feature_map)
            ScoreMaps_regr.append(ScoreMap)

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count,
                                 nms_threshold=config.RPN_NMS_THRESHOLD,
                                 name="ROI",
                                 anchors=self.anchors,
                                 config=config)([rpn_class, rpn_bbox])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            _, _, _, active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x))(input_image_meta)

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    rpn_rois, input_gt_class_ids, gt_boxes])

            # size = [batch, num_rois, class_num]
            classify_vote = VotePooling(config.TRAIN_ROIS_PER_IMAGE, config.C, config.K, config.POOL_SIZE, config.BATCH_SIZE, config.IMAGE_SHAPE, name="classify_vote")([rois] + ScoreMaps_classify)
            classify_output = KL.TimeDistributed(KL.Activation('softmax'),name="classify_output")(classify_vote)

            # 4 k^2 rather than 4k^2*C
            regr_vote = VotePooling(config.TRAIN_ROIS_PER_IMAGE, 4, config.K, config.POOL_SIZE, config.BATCH_SIZE, config.IMAGE_SHAPE, name="regr_vote")([rois] + ScoreMaps_regr)
            regr_output = KL.TimeDistributed(KL.Activation('linear'),name="regr_output")(regr_vote)

            rpn_class_loss = KL.Lambda(lambda x: KerasRFCN.Losses.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: KerasRFCN.Losses.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: KerasRFCN.Losses.mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, classify_vote, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: KerasRFCN.Losses.mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, regr_output])

            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes]

            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       classify_vote, classify_output, regr_output,
                       rpn_rois, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss]

            self.keras_model = KM.Model(inputs, outputs, name='rfcn_train')
        else: # inference

            # Network Heads
            # Proposal classifier and BBox regressor heads
            # size = [batch, num_rois, class_num]
            classify_vote = VotePooling(proposal_count, config.C, config.K, config.POOL_SIZE, config.BATCH_SIZE, config.IMAGE_SHAPE, name="classify_vote")([rpn_rois] + ScoreMaps_classify)
            classify_output = KL.TimeDistributed(KL.Activation('softmax'),name="classify_output")(classify_vote)

            # 4 k^2 rather than 4k^2*C
            regr_vote = VotePooling(proposal_count, 4, config.K, config.POOL_SIZE, config.BATCH_SIZE, config.IMAGE_SHAPE, name="regr_vote")([rpn_rois] + ScoreMaps_regr)
            regr_output = KL.TimeDistributed(KL.Activation('linear'),name="regr_output")(regr_vote)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, score)] in image coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, classify_output, regr_output, input_image_meta])

            self.keras_model = KM.Model([input_image, input_image_meta],
                             [detections, classify_output, regr_output, rpn_rois, rpn_class, rpn_bbox],
                             name='rfcn_inference')
            
    def build_rpn_model(self, anchor_stride, anchors_per_location, depth):
        """Builds a Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared
        weights.

        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).
        depth: Depth of the backbone feature map.

        Returns a Keras Model object. The model outputs, when called, are:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                    applied to anchors.
        """
        input_feature_map = KL.Input(shape=[None, None, depth],
                                     name="input_rpn_feature_map")
        outputs = self.rpn(input_feature_map, anchors_per_location, anchor_stride)
        return KM.Model([input_feature_map], outputs, name="rpn_model")

    def rpn(self, feature_map, anchors_per_location, anchor_stride):
        """Builds a Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared
        weights.

        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).
        depth: Depth of the backbone feature map.

        Returns a Keras Model object. The model outputs, when called, are:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                    applied to anchors.
        """

        shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                           strides=anchor_stride,
                           name='rpn_conv_shared')(feature_map)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                      activation='linear', name='rpn_class_raw')(shared)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = KL.Lambda(
            lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

        # Softmax on last dimension of BG/FG.
        rpn_probs = KL.Activation(
            "softmax", name="rpn_class_xxx")(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location, depth]
        # where depth is [x, y, log(w), log(h)]
        x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                      activation='linear', name='rpn_bbox_pred')(shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

        return rpn_class_logits, rpn_probs, rpn_bbox


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, anchors,
                 config=None, **kwargs):
        """
        anchors: [N, (y1, x1, y2, x2)] anchors defined in image coordinates
        """
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.anchors = anchors.astype(np.float32)

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Base anchors
        anchors = self.anchors

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = min(6000, self.anchors.shape[0])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = KerasRFCN.Utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = KerasRFCN.Utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        anchors = KerasRFCN.Utils.batch_slice(ix, lambda x: tf.gather(anchors, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = KerasRFCN.Utils.batch_slice([anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
        height, width = self.config.IMAGE_SHAPE[:2]
        window = np.array([0, 0, height, width]).astype(np.float32)
        boxes = KerasRFCN.Utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Normalize dimensions to range of 0 to 1.
        normalized_boxes = boxes / np.array([[height, width, height, width]])

        # Non-max suppression
        def nms(normalized_boxes, scores):
            indices = tf.image.non_max_suppression(
                normalized_boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(normalized_boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = KerasRFCN.Utils.batch_slice([normalized_boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)

############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas for each.

    Inputs:
    proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
            Class-specific bbox refinments.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)

    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [anchors, crowds]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine postive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = KerasRFCN.Utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])

    return rois, roi_gt_class_ids, deltas

def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinment, class_ids for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox"]
        outputs = KerasRFCN.Utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes],
            lambda w, x, y: detection_targets_graph(
                w, x, y, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, 1),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
        ]

############################################################
#  ROI pooling on Muti Bins
############################################################

def log2_graph(x):
    """Implementatin of Log2. TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)

class VotePooling(KE.Layer):
    def __init__(self, num_rois, channel_num, k, pool_shape, batch_size, image_shape, **kwargs):
        super(VotePooling, self).__init__(**kwargs)
        self.channel_num = channel_num
        self.k = k
        self.num_rois = num_rois
        self.pool_shape = pool_shape
        self.batch_size = batch_size
        self.image_shape = image_shape

    def call(self, inputs):
        boxes = inputs[0]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        score_maps = inputs[1:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(
            self.image_shape[0] * self.image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                score_maps[i], level_boxes, box_indices, [self.pool_shape * self.k, self.pool_shape * self.k],
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # position-sensitive ROI pooling + classify
        score_map_bins = []
        for channel_step in range(self.k*self.k):
            bin_x = K.variable( int(channel_step % self.k) * self.pool_shape, dtype='int32')
            bin_y = K.variable( int(channel_step / self.k) * self.pool_shape, dtype='int32')
            channel_indices = K.variable(list(range(channel_step*self.channel_num, (channel_step+1)*self.channel_num)), dtype='int32')
            croped = tf.image.crop_to_bounding_box(
                tf.gather( pooled, indices=channel_indices, axis=-1), bin_y, bin_x, self.pool_shape, self.pool_shape)
            # [pool_shape, pool_shape, channel_num] ==> [1,1,channel_num] ==> [1, channel_num]
            croped_mean = K.pool2d(croped, (self.pool_shape, self.pool_shape), strides=(1, 1), padding='valid', data_format="channels_last", pool_mode='avg')
            # [batch * num_rois, 1,1,channel_num] ==> [batch * num_rois, 1, channel_num] 
            croped_mean = K.squeeze(croped_mean, axis=1)
            score_map_bins.append(croped_mean)
        # [batch * num_rois, k^2, channel_num]
        score_map_bins = tf.concat(score_map_bins, axis=1)
        # [batch * num_rois, k*k, channel_num] ==> [batch * num_rois,channel_num]
        # because "keepdims=False", the axis 1 will not keep. else will be [batch * num_rois,1,channel_num]
        pooled = K.sum(score_map_bins, axis=1)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)
        
        return pooled

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.channel_num

############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
    return boxes


def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        deltas: [N, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """
    # Class IDs per ROI

    class_ids = np.argmax(probs, axis=1)
    # Class probability of the top class of each ROI
    class_scores = probs[np.arange(class_ids.shape[0]), class_ids]
    # Class-specific bounding box deltas
    deltas_specific = deltas[np.arange(deltas.shape[0])]
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = KerasRFCN.Utils.apply_box_deltas(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Convert coordiates to image domain
    # TODO: better to keep them normalized until later
    height, width = config.IMAGE_SHAPE[:2]
    refined_rois *= np.array([height, width, height, width])
    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)
    # Round and cast to int since we're deadling with pixels now
    refined_rois = np.rint(refined_rois).astype(np.int32)

    # Filter out background boxes
    keep = np.where(class_ids > 0)[0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep = np.intersect1d(
            keep, np.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[0])
    
    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores = class_scores[keep]
    pre_nms_rois = refined_rois[keep]
    
    nms_keep = []
    for class_id in np.unique(pre_nms_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        # Apply NMS
        class_keep = KerasRFCN.Utils.non_max_suppression(
            pre_nms_rois[ixs], pre_nms_scores[ixs],
            config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
    keep = np.intersect1d(keep, nms_keep).astype(np.int32)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = np.argsort(class_scores[keep])[::-1][:roi_count]
    keep = keep[top_ids]

    # Arrange output as [N, (y1, x1, y2, x2, class_ids, score)]
    # Coordinates are in image domain.
    result = np.hstack((refined_rois[keep],
                        class_ids[keep][..., np.newaxis],
                        class_scores[keep][..., np.newaxis]))
    return result

def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]

# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids

class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        def wrapper(rois, mrcnn_class, mrcnn_bbox, image_meta):
            detections_batch = []
            _, _, window, _ = parse_image_meta(image_meta)
            for b in range(self.config.BATCH_SIZE):
                detections = refine_detections(
                    rois[b], mrcnn_class[b], mrcnn_bbox[b], window[b], self.config)
                # Pad with zeros if detections < DETECTION_MAX_INSTANCES
                gap = self.config.DETECTION_MAX_INSTANCES - detections.shape[0]
                assert gap >= 0
                if gap > 0:
                    detections = np.pad(
                        detections, [(0, gap), (0, 0)], 'constant', constant_values=0)
                detections_batch.append(detections)

            # Stack detections and cast to float32
            # TODO: track where float64 is introduced
            detections_batch = np.array(detections_batch).astype(np.float32)
            # Reshape output
            # [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
            return np.reshape(detections_batch, [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

        # Return wrapped function
        return tf.py_func(wrapper, inputs, tf.float32)

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)
