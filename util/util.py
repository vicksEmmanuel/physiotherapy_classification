from torchvision.transforms.functional import resize
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.transforms._functional_video import normalize
from util.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
import numpy as np
import torch
from torchvision.transforms._functional_video import normalize
from torch.utils.data import DataLoader,random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 

import cv2
from util.PackPathwayTransform import PackPathway
from util.config import *
from util.action_dataset import ActionDataset



def single_transformer():
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    return Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway()
            ]
        )

def ava_inference_transform(
    clip,
    boxes,
    num_frames = 4, #if using slowfast_r50_detection, change this to 32
    crop_size = 256,
    data_mean = [0.45, 0.45, 0.45],
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = None, #if using slowfast_r50_detection, change this to 4
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]

    
    print(f"Clip {clip}")

    return clip, torch.from_numpy(boxes), ori_boxes

def adjust_boxes(boxes, original_height, original_width, new_height, new_width):
    # Calculate scale factors for width and height
    width_scale = new_width / original_width
    height_scale = new_height / original_height

    # Adjust box coordinates
    adjusted_boxes = boxes.copy()
    adjusted_boxes[:, 0] = boxes[:, 0] * width_scale  # x1
    adjusted_boxes[:, 2] = boxes[:, 2] * width_scale  # x2
    adjusted_boxes[:, 1] = boxes[:, 1] * height_scale  # y1
    adjusted_boxes[:, 3] = boxes[:, 3] * height_scale  # y2

    return adjusted_boxes

def ava_inference_transform2(sample_dict, num_frames=4, slow_fast_alpha=None, crop_size=256, data_mean=[0.45, 0.45, 0.45], data_std=[0.225, 0.225, 0.225]):
    clip = sample_dict["video"]
    boxes = np.array(sample_dict.get("boxes", []))
    ori_boxes = boxes.copy()


    clip = resize(clip, (crop_size, crop_size))
    boxes = adjust_boxes(boxes, clip.shape[2], clip.shape[3], crop_size, crop_size)

   # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)
    # boxes = torch.cat([torch.zeros(boxes.shape[0],1), boxes], dim=1)


    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )

    num_boxes = boxes.shape[0]
    dummy_labels = np.zeros((num_boxes, 1))
    boxes_with_labels = np.hstack((boxes, dummy_labels))

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, int(clip.shape[1] // slow_fast_alpha)
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]
    

    # boxes = torch.cat([torch.zeros(boxes.shape[0],1), boxes], dim=1)

    # Update sample_dict with transformed data
    transformed_sample_dict = sample_dict.copy()
    transformed_sample_dict["video"] = clip

    if len(boxes) > 0:
        transformed_sample_dict["boxes"] = torch.from_numpy(boxes_with_labels).float()
    transformed_sample_dict["ori_boxes"] = torch.from_numpy(boxes).float()
    
    return transformed_sample_dict
