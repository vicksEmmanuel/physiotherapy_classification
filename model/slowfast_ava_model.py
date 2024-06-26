from pytorchvideo.models.hub import slow_r50_detection, slowfast_r50_detection
from pytorchvideo.models.resnet import create_resnet_with_roi_head
from torch import nn
from pytorch_lightning import LightningModule
import torch
from torch.nn import functional as F
from torchmetrics.functional import accuracy

import torch
import torch.nn as nn
from pytorchvideo.models.resnet import create_resnet_with_roi_head

from util.actions import Action

class CustomSlowR50Detection(nn.Module):
    def __init__(self, pretrained=True, num_classes=len(Action().action)):
        super().__init__()
        self.base_model = slow_r50_detection(pretrained=pretrained)
        detection_head = self.base_model.detection_head
        num_features = detection_head.proj.in_features
        detection_head.proj = nn.Linear(num_features, num_classes)


    def forward(self, x, bboxes):
        return self.base_model(x, bboxes)


class SlowFastAva(LightningModule):
    def __init__(self, drop_prob=0.5, num_frames=4, num_classes=len(Action().action)):
        super().__init__()

        self.drop_prob = drop_prob
        self.num_classes = len(Action().action)
        self.num_frames = num_frames
        self.save_hyperparameters()

        self.load()

    def load(self):
        self.model = CustomSlowR50Detection(pretrained=False, num_classes=self.num_classes)

    def forward(self, x, bboxes):
        return self.model(x, bboxes)

    def configure_optimizers(self):
        learning_rate = 1e-4
        optimizer = torch.optim.ASGD(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=4, cooldown=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}
    
    
    def on_training_epoch_end(self):
        sch = self.lr_schedulers()

        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["valid_loss"])
        else:
            sch.step()

    def training_step(self, batch, batch_idx):
        print("Training step")

        total_loss = 0
        total_acc = 0
        for batch_item in batch:
            print(f"Video Name: {batch_item['video_name']}")
            videos = batch_item['video']
            videos = videos.unsqueeze(0)
            bboxes = batch_item['boxes']

            labels = batch_item['labels']
            labels = self._shared_label_process(labels, num_classes=self.num_classes)
            print(f"Videos shape: {videos.shape} Bboxes shape: {bboxes.shape}  Labels shape: {labels} : shape {labels.shape}")

            outputs = self(videos, bboxes)

            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            acc = accuracy(outputs.softmax(dim=-1), labels,task="multiclass", num_classes=self.num_classes)

            total_loss += loss
            total_acc += acc

        avg_loss = total_loss / len(batch)
        avg_acc = total_acc / len(batch)

        metrics = {"train_acc": avg_acc, "train_loss": avg_loss}
        print(metrics)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        print("Validation step")

        total_loss = 0
        total_acc = 0
        for batch_item in batch:
            print(f"Video Name: {batch_item['video_name']}")
            videos = batch_item['video']
            videos = videos.unsqueeze(0)
            bboxes = batch_item['boxes']

            labels = batch_item['labels']
            labels = self._shared_label_process(labels, num_classes=self.num_classes)
            print(f"Videos shape: {videos.shape} Bboxes shape: {bboxes.shape}  Labels shape: {labels} : shape {labels.shape}")


            outputs = self(videos, bboxes)

            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            acc = accuracy(outputs.softmax(dim=-1), labels, task="multiclass", num_classes=self.num_classes)

            total_loss += loss
            total_acc += acc

        avg_loss = total_loss / len(batch)
        avg_acc = total_acc / len(batch)

        metrics = {"valid_loss": avg_loss, "valid_acc": avg_acc}
        print(metrics)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):

        total_loss = 0
        total_acc = 0
        for batch_item in batch:
            videos = batch_item['video']
            videos = videos.unsqueeze(0)
            bboxes = batch_item['boxes']
            
            labels = batch_item['labels']
            labels = self._shared_label_process(labels, num_classes=self.num_classes)
            print(f"Videos shape: {videos.shape} Bboxes shape: {bboxes.shape}  Labels shape: {labels} : shape {labels.shape}")


            outputs = self(videos, bboxes)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            acc = accuracy(outputs.softmax(dim=-1), labels, task="multiclass", num_classes=self.num_classes)

            total_loss += loss
            total_acc += acc

        avg_loss = total_loss / len(batch)
        avg_acc = total_acc / len(batch)
        
        metrics = {"test_acc": avg_acc, "test_loss": avg_loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    def _shared_label_process(self, labels_list, num_classes=len(Action().action)):
        labels = torch.zeros((len(labels_list), num_classes), dtype=torch.float, device=self.device)
    
        for idx, class_indices in enumerate(labels_list):
            labels[idx, [x - 1 for x in class_indices]] = 1.0
        
        return labels
