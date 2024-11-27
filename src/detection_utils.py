from __future__ import absolute_import, division, print_function
import numpy as np
import cv2
import torch
import os
from typing import Any

import thirdparty.detector._init_paths
from model.utils.config import cfg
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.resnet import resnet


def load_faster_rcnn() -> Any:
    """
    Initializes and returns a pre-trained Faster R-CNN model for egoecentric hand-object detection.

    :return: The initialized Faster R-CNN model, ready for inference.
    """
    load_name = os.path.join(
        "thirdparty/detector/models/res101_handobj_100K/pascal_voc",
        'faster_rcnn_1_8_132028.pth'
    )
    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()
    
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('Loaded hand-object-detector successfully!')
    
    fasterRCNN.cuda()
    fasterRCNN.eval()
    return fasterRCNN

def _get_image_blob(im: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts an image into a network input.
    Arguments:
    im (ndarray): a color image in BGR order
    Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def hand_object_detection(
    im: np.ndarray,
    fasterRCNN: Any,
    thresh_hand: float = 0.5,
    thresh_obj: float = 0.5
) -> tuple[list, list]:
    """
    Detects hands and objects in an image using a Faster R-CNN model.
    More information about the model can be found in the thirdpary module: https://github.com/ddshan/hand_object_detector 

    :param im: Input image as a numpy array.
    :param fasterRCNN: Pre-trained Faster R-CNN model used for detection.
    :param thresh_hand: Confidence threshold for detecting hands. Defaults to 0.5.
    :param thresh_obj: Confidence threshold for detecting objects. Defaults to 0.5.
    :return: Tuple containing:
        - hand_detections: List of detected hand bounding boxes or detection data.
        - object_detections: List of detected object bounding boxes or detection data.
    """
    im_data = torch.FloatTensor(1).cuda()
    im_info = torch.FloatTensor(1).cuda()
    num_boxes = torch.LongTensor(1).cuda()
    gt_boxes = torch.FloatTensor(1).cuda()
    box_info = torch.FloatTensor(1)


    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()
        box_info.resize_(1, 1, 5).zero_()

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    # Extract predicted parameters
    contact_vector = loss_list[0][0]  # Hand contact state info
    offset_vector = loss_list[1][0].detach()  # Offset vector
    lr_vector = loss_list[2][0].detach()  # Hand side info

    # Get hand contact 
    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

    # Get hand side 
    lr = torch.sigmoid(lr_vector) > 0.5
    lr = lr.squeeze(0).float()

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4 * len(fasterRCNN.classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    obj_dets, hand_dets = None, None
    for j in range(1, len(fasterRCNN.classes)):
        if fasterRCNN.classes[j] == 'hand':
            inds = torch.nonzero(scores[:,j] > thresh_hand).view(-1)
        elif fasterRCNN.classes[j] == 'targetobject':
            inds = torch.nonzero(scores[:,j] > thresh_obj).view(-1)

        if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if fasterRCNN.classes[j] == 'targetobject':
                obj_dets = cls_dets.cpu().numpy()
            if fasterRCNN.classes[j] == 'hand':
                hand_dets = cls_dets.cpu().numpy()

    return hand_dets, obj_dets