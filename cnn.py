import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import random
import kagglehub
from sklearn.model_selection import train_test_split

dataset_path = '/kaggle/input/udacity-self-driving-car-dataset/data/export'
annotations_path = '/kaggle/input/udacity-self-driving-car-dataset/data/export/_annotations.csv'
annotations = pd.read_csv(annotations_path)
img_files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith('jpg')]

for i in range (3):
    img = Image.open(img_files[i])
    plt.figure()
    plt.imshow(img)
    plt.title(f"Image {i+1}")
    plt.axis('off')
    plt.show()
    print(annotations.head)

#function to calculate intersection over union; eliminate possible duplicate predictions of overlapping bounding boxes
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    #compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    #compute area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    unionArea = boxAArea + boxBArea - interArea
    
    return interArea / unionArea if unionArea != 0 else 0

#remove duplicate bounding boxes based on IoU threshold
def filter_duplicates(boxes, threshold=0.5):
    filtered_boxes = []
    for i, boxA in enumerate(boxes):
        to_add = True
        for boxB in filtered_boxes:
            if iou(boxA, boxB) > threshold:
                to_add = False
                break
        if to_add:
            filtered_boxes.append(boxA)
    return filtered_boxes

#example bounding boxes
bounding_boxes = [
    [100, 150, 200, 250],
    [110, 160, 210, 260],
    [300, 350, 400, 450],
]
filtered_boxes = filter_duplicates(bounding_boxes)
#print(filtered_boxes)

import cv2

#load annotations
annotations = pd.read_csv(annotations_path)

#group annotations by filename
grouped_annotations = annotations.groupby("filename")
def draw_bounding_boxes(image_file, boxes, save_image=False, output_path=None):
    image_path = os.path.join(dataset_path, image_file)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    #draw each bounding box
    for _, row in boxes.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['class']

        #draw rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Put label text
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if save_image:
        # Save image with bounding boxes
        if output_path:
            cv2.imwrite(output_path, image)  # Save to given output path
        else:
            cv2.imwrite('output.jpg', image)  # Default save

    #convert BGR to RGB for displaying
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()

#check random file's bounding box
image_file = "1478900859981702684_jpg.rf.6830635c7d9197475638f0818f5dd103.jpg"
boxes = grouped_annotations.get_group(image_file)  # Retrieve bounding boxes for the image
draw_bounding_boxes(image_file, boxes)

def normalize_bounding_boxes(bboxes, image_width, image_height):
    normalized_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        
        #normalize bounding box coordinates
        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        normalized_bboxes.append([x_center, y_center, width, height])
    
    return normalized_bboxes

#check example
image_width = 512
image_height = 512
bounding_boxes = [
    [100, 150, 200, 250],  #xmin, ymin, xmax, ymax
    [110, 160, 210, 260]
]

#filter duplicates (IoU threshold = 0.5)
filtered_boxes = filter_duplicates(bounding_boxes)

#normalize bounding boxes
image_width = 512
image_height = 512
normalized_boxes = normalize_bounding_boxes(filtered_boxes, image_width, image_height)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_files, annotations, transform=None):
        self.img_files = img_files
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        image = Image.open(img_file)
        image = image.convert("RGB")  # Make sure image is in RGB

        # Get bounding boxes for the image
        filename = os.path.basename(img_file)
        boxes = self.annotations.get_group(filename)

        # Normalize bounding boxes
        bboxes = boxes[['xmin', 'ymin', 'xmax', 'ymax']].values
        image_width, image_height = image.size
        normalized_bboxes = normalize_bounding_boxes(bboxes, image_width, image_height)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(normalized_bboxes, dtype=torch.float32)
