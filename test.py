import random

from matplotlib import patches, pyplot as plt
from util.get_audio import get_audio
from PIL import Image, ImageDraw
import os
from torchvision.transforms.functional import normalize as normalize_image
from torchvision.transforms.functional import to_tensor
from util.actions import Action
from detectron2.config import get_cfg
from pytorchvideo.models.hub import slow_r50_detection 
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from util.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
import torch
from torch.nn import functional as F
from torch import nn
from torchmetrics.functional import accuracy
from torchvision.transforms._functional_video import normalize
import numpy as np
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


video_path = 'test.mp4'
new_path = get_video_clip_and_resize(video_path)
encoded_vid = EncodedVideo.from_path(new_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = slow_r50_detection(True)
model = SlowFastAva.load_from_checkpoint("checkpoints/last.ckpt")
print(model)

model.eval()
model.to(device)

actions = Action().action
label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('data/actions_dataset/activity_net.pbtxt')

print("Label map: ", label_map)
print("Allowed class ids: ", allowed_class_ids)

actions_per_second = []

video_visualizer = VideoVisualizer(
    num_classes=len(actions) + 1,
    class_names_path='data/actions_dataset/activity_net.pbtxt',
    top_k=3, 
    mode="thres",
    thres=0.4
)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


def get_single_bboxes(inp_imgs):
    _, height, width = inp_imgs.shape[1:]
    x1 = random.randint(0, max(0, width - 100))
    y1 = random.randint(0, max(0, height - 100))
    x2 = min(x1 + random.randint(100, 200), width)
    y2 = min(y1 + random.randint(100, 200), height)

    # Calculate the center of the bounding box
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0

    # Calculate the width and height of the bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Normalize the center coordinates and dimensions
    normalized_x_center = x_center / width
    normalized_y_center = y_center / height
    normalized_width = bbox_width / width
    normalized_height = bbox_height / height

    # Format the bounding box coordinates
    x1 = float(f"{normalized_x_center:.4f}")
    y1 = float(f"{normalized_y_center:.4f}")
    x2 = float(f"{normalized_width:.4f}")
    y2 = float(f"{normalized_height:.4f}")
    predicted_boxes = torch.tensor([[x1, y1, x2, y2]])
    return predicted_boxes


def get_person_bboxes(inp_img, predictor):
    predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    predicted_boxes = boxes[np.logical_and(classes==0, scores>0.75 )].tensor.cpu() # only person
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
    gif_imgs = []
    confidence_threshold = 0.3
    actions_per_second = []
    total_duration = int(encoded_vid.duration)  # Total duration in seconds
    audio_timestamps =  get_audio(video_path, total_duration=total_duration)

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
        # predicted_boxes = get_single_bboxes(inp_imgs)
        # plot_bounding_boxes(inp_imgs, inp_img, predicted_boxes)

        predicted_boxes = get_person_bboxes(inp_img, predictor)

        
        if len(predicted_boxes) == 0:
            print(f"No person detected in second {start_sec} - {end_sec}.")
            continue

        print(f"Predicted boxes for second {start_sec} - {end_sec}: {predicted_boxes.numpy()}")

        inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy())

        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
        inputs = inputs.unsqueeze(0)

        # Generate actions predictions for the bounding boxes in the clip.
        # The model here takes in the pre-processed video clip and the detected bounding boxes.
        preds = model(inputs.to(device), inp_boxes.to(device))

        # print(f"Predictions for second {start_sec} - {end_sec}: {preds}")

        preds= preds.to('cpu')
        # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
        preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)

        append_to_actions_list(preds, start_sec, end_sec, confidence_threshold, audio_timestamps[i], actions_per_second)

        # Plot predictions on the video and save for later visualization.
        inp_imgs = inp_imgs.permute(1,2,3,0)
        inp_imgs = inp_imgs/255.0
        out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds, predicted_boxes)
        gif_imgs += out_img_pred
    
    print("Finished generating predictions.")


    height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]

    vide_save_path = 'output.mp4'
    video = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'DIVX'), 7, (width,height))

    for image in gif_imgs:
        img = (255*image).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)
    video.release()

    print('Predictions are saved to the video file: ', vide_save_path)
    return actions_per_second


def append_to_actions_list(preds, start_sec, end_sec, confidence_threshold=0.5, audio_timestamps=None, actions_per_sec=[]):
    # Get the predicted actions and their probabilities
    predicted_actions = preds.argmax(dim=1).tolist()
    predicted_probs = preds.max(dim=1).values.tolist()

    # Filter the actions based on the confidence threshold
    filtered_actions = [actions[action_id] for action_id, prob in zip(predicted_actions, predicted_probs) if prob >= confidence_threshold]
    print(f"Predicted actions for second {start_sec} - {end_sec}: {filtered_actions}")
    
    actions_per_sec.append({
        "start": start_sec,
        "end": end_sec,
        "actions": filtered_actions,
        "audio": audio_timestamps
    })

all_actions = generate_actions_from_video(video_path)

print("All actions: ", all_actions)