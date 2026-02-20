import os
import copy
import random
import json
import yaml
import glob
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
from   zipfile import ZipFile
import argparse
from PIL import Image
import PIL.Image
import shutil
from IPython.display import Image
from sklearn.model_selection import train_test_split
 
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms as T
 
from ultralytics import YOLO

model = YOLO("yolov12s.yaml") #build a model from YAML

with open("yolov12s_dataset/data.yaml",'r') as stream:
     num_classes = str(yaml.safe_load(stream)['nc'])
     
#Define a project --> Destination directory for all results
project = "yolov12s_dataset/results"
#Define subdirectory for this specific training
name = "300_epochs-v12s"
ABS_PATH = os.getcwd()

#Train the model
results = model.train(
    data = os.path.join(ABS_PATH, "yolov12s_dataset/data.yaml"),
    project = project,
    name = name,
    epochs = 300,
    mosaic = 1.0,
    scale = 0.5,
    patience = 0 , #setting patience=0 to disable early stopping,
    flipup = 0.5,
    fliplr = 0.5,
    degrees = 90.0,
    batch = 4,
    imgsz=1216
)
