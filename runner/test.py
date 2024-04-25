import random

from matplotlib import patches, pyplot as plt
from PIL import Image, ImageDraw
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from torchvision.transforms.functional import normalize as normalize_image
from torchvision.transforms.functional import to_tensor
from util.actions import Action
from detectron2.config import get_cfg
from pytorchvideo.models.hub import slow_r50_detection 
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import torch
from torch.nn import functional as F
from torch import nn
from torchmetrics.functional import accuracy
from torchvision.transforms._functional_video import normalize
import numpy as np
from util.get_audio import get_audio
from util.util_2 import  get_video_clip_and_resize, get_video_clip_and_resize3, get_video_clip_and_resize2 # Ensure this import matches your project structure
from pytorchvideo.data.encoded_video import EncodedVideo
import torch
from torchvision.transforms import functional as F
from util.video_visualizer import VideoVisualizer
import pytorchvideo.models.slowfast as SlowFastModel
import cv2
from model.slowfast_ava_model import SlowFastAva  # Ensure this import matches your project structure
from util.util import single_transformer,ava_inference_transform
from pytorchvideo.models.resnet import create_resnet, create_resnet_with_roi_head
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
import json

actions = Action().action
threshold = 0.5

def get_bboxes(inp_imgs, num_boxes):
    _, height, width = inp_imgs.shape[1:]
    bboxes = []

    # Generate random bounding boxes
    for _ in range(num_boxes):
        # Generate random width and height for the bounding box
        box_width = random.randint(50, 200)
        box_height = random.randint(50, 200)
        
        # Generate random coordinates for the top-left corner of the bounding box
        x1 = random.uniform(0, width - box_width)
        y1 = random.uniform(0, height - box_height)
        
        # Calculate the coordinates for the bottom-right corner of the bounding box
        x2 = x1 + box_width
        y2 = y1 + box_height
        
        bboxes.append([x1, y1, x2, y2])

    # Generate structured bounding boxes - split height into 4s
    for i in range(4):
        y1 = i * (height / 4)
        y2 = (i + 1) * (height / 4)
        x1 = 0
        x2 = width
        bboxes.append([x1, y1, x2, y2])

    # Generate structured bounding boxes - split height into 2s
    for i in range(2):
        y1 = i * (height / 2)
        y2 = (i + 1) * (height / 2)
        x1 = 0
        x2 = width
        bboxes.append([x1, y1, x2, y2])

    # Generate a bounding box covering the entire video
    bboxes.append([0, 0, width, height])

    # Generate structured bounding boxes - split width into 4s
    for i in range(4):
        x1 = i * (width / 4)
        x2 = (i + 1) * (width / 4)
        y1 = 0
        y2 = height
        bboxes.append([x1, y1, x2, y2])

    # Generate structured bounding boxes - split width into 2s
    for i in range(2):
        x1 = i * (width / 2)
        x2 = (i + 1) * (width / 2)
        y1 = 0
        y2 = height
        bboxes.append([x1, y1, x2, y2])

    predicted_boxes = torch.tensor(bboxes, dtype=torch.float32)
    return predicted_boxes

def plot_bounding_boxes(inp_imgs, inp_img, predicted_boxes):
    frame = inp_imgs[0]  # Access the first video in the batch
    frame = frame[0, :, :].detach().cpu().numpy()
    frame = (frame - frame.min()) / (frame.max() - frame.min())

    fig, axes = plt.subplots(1, 1, figsize=(12, 12))
    axes.imshow(frame)
    axes.axis('off')
    H, W = inp_img.shape[:2]

    for box in predicted_boxes:
        x_center, y_center, width, height = box
        x_min = (x_center - width / 2) * W
        y_min = (y_center - height / 2) * H
        x_max = (x_center + width / 2) * W
        y_max = (y_center + height / 2) * H
        rect = patches.Rectangle((x_min, y_min), width * W, height * H, linewidth=1, edgecolor='r', facecolor='none')
        axes.add_patch(rect)

        # Display the label above the bounding box
        axes.text(x_min, y_min - 5, "", color='r', fontsize=8, verticalalignment='top')

    plt.show()

def generate_actions_from_video(video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = slow_r50_detection(True)
    model = SlowFastAva.load_from_checkpoint("checkpoints/last.ckpt")
    model.eval()
    model.to(device)

    video_visualizer = VideoVisualizer(
        num_classes=len(actions),
        class_names_path='data/actions_dataset/activity_net.pbtxt',
        top_k=3, 
        mode="thres",
        thres=threshold
    )

    path_without_extension = os.path.splitext(video_path)[0]
    new_path = f"{path_without_extension}_resized.mp4"

    if not os.path.exists(new_path):
        new_path = get_video_clip_and_resize(video_path)


    encoded_vid = EncodedVideo.from_path(new_path)
    gif_imgs = []
    confidence_threshold = threshold
    actions_per_second = []
    total_duration = int(encoded_vid.duration)  # Total duration in seconds
    audio_path = f"{path_without_extension}.wav"

    audio_timestamps =  get_audio(
        video_path, 
        total_duration=total_duration, 
        speech_file_path=audio_path
    )

    for i in range(0,len(audio_timestamps)):
        start_sec = audio_timestamps[i]["start"]
        end_sec = audio_timestamps[i]["end"]

        # Generate clip around the designated time stamps
        inp_imgs = encoded_vid.get_clip(start_sec=start_sec, end_sec=end_sec)
        inp_imgs = inp_imgs['video']
        

        # Generate people bbox predictions using Detectron2's off the self pre-trained predictor
        # We use the the middle image in each clip to generate the bounding boxes.
        inp_img = inp_imgs[:,inp_imgs.shape[1]//2,:,:]
        inp_img = inp_img.permute(1,2,0)

        # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
        predicted_boxes = get_bboxes(inp_imgs, 10)

        if len(predicted_boxes) == 0:
            print(f"No person detected in second {start_sec} - {end_sec}.")
            continue

        inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy())

        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
        inputs = inputs.unsqueeze(0)

        # Generate actions predictions for the bounding boxes in the clip.
        # The model here takes in the pre-processed video clip and the detected bounding boxes.
        preds = model(inputs.to(device), inp_boxes.to(device))

        preds= preds.to('cpu')
        append_to_actions_list(preds, start_sec, end_sec, confidence_threshold, audio_timestamps[i], actions_per_second)

        # Plot predictions on the video and save for later visualization.
        inp_imgs = inp_imgs.permute(1,2,3,0)
        inp_imgs = inp_imgs/255.0
        out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds, predicted_boxes)
        gif_imgs += out_img_pred
    
    print("Finished generating predictions.")
    # TODO: Generate videos that contains the actions
    # height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]
    # vide_save_path = 'output.mp4'
    # video = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'DIVX'), 7, (width,height))

    # for image in gif_imgs:
    #     img = (255*image).astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     video.write(img)
    # video.release()

    # print('Predictions are saved to the video file: ', vide_save_path)

    
    os.remove(new_path)
    return actions_per_second

def append_to_actions_list(preds, start_sec, end_sec, confidence_threshold=0.5, audio_timestamps=None, actions_per_sec=[]):
    # Get the predicted actions and their probabilities
    predicted_actions = preds.argmax(dim=1).tolist()
    predicted_probs = preds.max(dim=1).values.tolist()

    # Filter the actions based on the confidence threshold
    filtered_actions = [actions[action_id] for action_id, prob in zip(predicted_actions, predicted_probs) if prob >= confidence_threshold]
    actions_pred = list(set(filtered_actions))

    actions_per_sec.append({
        "actions": actions_pred,
        "audio": audio_timestamps,
        "discussions": audio_timestamps["text"],
        "actions_and_discussions": {
            "actions": actions_pred,
            "discussions": audio_timestamps["text"],
            "start_time": start_sec,
            "end_time": end_sec
        }
    })

def get_new_data(all_data):
    actions = [x["actions"] for x in all_data]
    flattened_actions = [action for sublist in actions for action in sublist]

    new_data = {
        "actions":  list(set(flattened_actions)),
        "discussions": [x["discussions"] for x in all_data],
        "actions_and_discussions": [x["actions_and_discussions"] for x in all_data]
    }
    return new_data


def get_new_data_from_video(video_path):
    all_data = generate_actions_from_video(video_path)
    return get_new_data(all_data)


def set_videos(data: object , grade, video_paths: list):
    all_data = []
    for video_path in video_paths:
        all_data.append(get_new_data_from_video(video_path))
    data[grade]  = all_data


def list_of_data_generated(data):
    good = []
    brief = []
    average = []
    ends_width = ('.MOV', '.mp4', '.webp', '.avi', '.mkv')

    for obj in os.listdir('data/test_data'):
        if obj == 'good':
            good = [f'data/test_data/good/{x}' for x in os.listdir(f'data/test_data/good') if x.endswith(ends_width)]
        elif obj == 'brief':
            brief = [f'data/test_data/brief/{x}' for x in os.listdir(f'data/test_data/brief') if x.endswith(ends_width)]
        elif obj == 'average':
            average = [f'data/test_data/average/{x}' for x in os.listdir(f'data/test_data/average') if x.endswith(ends_width)]

    data = {
        "good": good,
        "brief": brief,
        "average": average
    }

    for obj in data:
        set_videos(data, obj, data[obj])

    return data



if __name__ == '__main__':
    # Generate data
    data = list_of_data_generated({})
    # Convert data to JSON format
    json_data = json.dumps(data)

    # Write JSON data to a file
    with open('dellma/data/grades/report.json', 'w') as file:
        file.write(json_data)
    

