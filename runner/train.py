

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning import Trainer, seed_everything
from util.action_dataset import ActionDataset
from util.config import CFG
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from util.PackPathwayTransform import PackPathway
from torch.utils.data import Dataset

from util.actions import Action
from model.slowfast_ava_model import SlowFastAva
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths

from util.load_dataloader import prepare_ava_dataset


def train(config):


    print("Training begins:")

    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('data/actions_dataset/activity_net.pbtxt')
    action_ids = list(label_map.keys())
    print(f"Action IDs: {action_ids}")
    length_of_actions = len(action_ids)
    print(f"Length of actions:  {length_of_actions}")
    action_ids, length_of_actions

    print(f"Length of actions:  {length_of_actions}")


    model = SlowFastAva(
        drop_prob=config.drop_prob, 
        num_frames=config.num_frames,
        num_classes=length_of_actions
    )

    print(f"Model: {model}")

    loaders = {
        p: prepare_ava_dataset(p,  config=config)
            for p in [ 'train', 'val'] 
    }


    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/', # Directory where the checkpoints will be saved
        filename='{epoch}-{val_loss:.2f}', # File name, which can include values from logging
        save_top_k=1, # Save the top 3 models according to the metric monitored below
        verbose=True,
        monitor='valid_loss', # Metric t o monitor for improvement
        mode='min', # Mode 'min' is for loss, 'max' for accuracy
        every_n_epochs=1, # Save checkpoint every epoch
        save_last=True, # Save the last model regardless of the monitored metric
        
    )

    trainer = Trainer(
        # logger=wandb_logger,
        # accelerator='cpu', # 'ddp' for distributed computing
        accelerator='gpu', # 'ddp' for distributed computing
        # devices=8, # Use 1 GPU
        # overfit_batches=0.05,
        # strategy='ddp',
        # sync_batchnorm=True,
        max_epochs=config.num_epochs,
        num_sanity_val_steps=0,
        limit_train_batches=config.limit_step_per_batch,
        limit_val_batches=config.limit_step_per_batch,
        limit_test_batches=config.limit_step_per_batch,    
        callbacks=[
            RichProgressBar(),
            checkpoint_callback
        ],
    )

    last_checkpoint = "checkpoints/last.ckpt"

    if os.path.exists(last_checkpoint):
        trainer.fit(model, loaders['train'], loaders['val'], ckpt_path=last_checkpoint)
        trainer.test(model, loaders['test'])
    else:
        trainer.fit(model, loaders['train'], loaders['val'])
        trainer.test(model, loaders['test'])

train(CFG)
    
