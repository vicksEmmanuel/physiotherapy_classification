from util.util import ava_inference_transform2
from torch.utils.data import IterableDataset
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from util.actions import Action
from pytorchvideo.data import Ava
import pandas as pd
import json
from pytorchvideo.data import Ava
from pytorchvideo.data.clip_sampling import make_clip_sampler
import numpy as np
from torch.utils.data import DataLoader,random_split
from util.config import CFG
from util.util import ava_inference_transform2
import os

from pytorchvideo.transforms.functional import (
    convert_to_one_hot,
)


def show_image(frame, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(frame)
    ax.axis('off')  # Hide the axis
    
    H, W = frame.shape[:2]  # Height and Width of the frame

    print(f"{H} {W}")

    for box in boxes:
        x_min, y_min, x_max, y_max = box
        # Scale the box coordinates to match the image dimensions
        rect = patches.Rectangle((x_min * W, y_min * H), (x_max - x_min) * W, (y_max - y_min) * H,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show(block=True)
    plt.pause(.1)


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


def draw_image(sample_1):
    print(sample_1)
    frame = sample_1['video'][0] # Access the first video in the batch

    frame = frame[0, :, :].detach().cpu().numpy()
    frame = (frame - frame.min()) / (frame.max() - frame.min())
    boxes = sample_1['ori_boxes']  # Retrieve bounding box data
    show_image(frame, boxes)


def plot_image(frame, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(frame)
    ax.axis('off')  # Hide the axis
    H, W = frame.shape[:2]  # Height and Width of the frame
    print(f"{H} {W}")
    
    for box in boxes:
        x_center, y_center, width, height = box
        x_min = (x_center - width / 2) * W
        y_min = (y_center - height / 2) * H
        x_max = (x_center + width / 2) * W
        y_max = (y_center + height / 2) * H
        rect = patches.Rectangle((x_min, y_min), width * W, height * H, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    return fig


def visualize_dataset(dataset, num_images=10):
    num_rows = 5
    num_cols = num_images // num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    
    # Flatten the axes array to handle different subplot dimensions
    axes = axes.flatten()
    
    for i in range(num_images):
        sample = next(dataset)
        print(sample)
        frame = sample['video'][0]  # Access the first video in the batch
        frame = frame[0, :, :].detach().cpu().numpy()
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        boxes = sample['ori_boxes']  # Retrieve bounding box data
        labels = sample['labels']  # Retrieve labels
        
        axes[i].imshow(frame)
        axes[i].axis('off')
        
        H, W = frame.shape[:2]  # Height and Width of the frame
        for box, label in zip(boxes, labels):
            x_center, y_center, width, height = box
            x_min = (x_center - width / 2) * W
            y_min = (y_center - height / 2) * H
            x_max = (x_center + width / 2) * W
            y_max = (y_center + height / 2) * H
            rect = patches.Rectangle((x_min, y_min), width * W, height * H, linewidth=1, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
            
            # Display the label above the bounding box
            axes[i].text(x_min, y_min - 5, str(label), color='r', fontsize=8, verticalalignment='top')
    
    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def visualize_ava_dataset(dataset):
    # All videos are of the form cthw and fps is 30
    # Clip is samples at time step = 2 secs in video
    sample_1 = next(dataset)
    draw_image(sample_1)


def prepare_ava_dataset(phase='train', config=CFG):

    def transform(sample_dict):
        return ava_inference_transform2(
            sample_dict, 
            num_frames = config.num_frames, 
            slow_fast_alpha = None,
            crop_size=256, 
            data_mean=[0.45, 0.45, 0.45], 
            data_std=[0.225, 0.225, 0.225]
        )


    prepared_frame_list = f"data/frames_dataset/frame_lists/{phase}.csv"
    frames_label_file_path = f"data/frames_dataset/annotations/{phase}.csv"
    label_map_path = "data/actions_dataset/activity_net.pbtxt"


    iterable_dataset = Ava(
        frame_paths_file=prepared_frame_list,
        frame_labels_file=frames_label_file_path,
        clip_sampler=make_clip_sampler("random", 20.0),
        label_map_file=label_map_path,
        transform=transform
    )

    # dataset = AvaDataset(iterable_dataset)

    loader = DataLoader(
        iterable_dataset, 
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        # num_batches_per_epoch=500, #@TODO: Evaluate this
        collate_fn=lambda x: x 
    )

    # Shows a picture of the first video in the dataset
    # visualize_ava_dataset(iterable_dataset)
    # visualize_dataset(iterable_dataset)

    return loader

class AvaDataset(IterableDataset):
    def __init__(self, iterable_dataset):
        self.data = []
        for item in iterable_dataset:
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]