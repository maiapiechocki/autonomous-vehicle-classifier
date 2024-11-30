Object Detection with PyTorch
This project implements an object detection model using PyTorch. The goal is to train a deep learning model to detect various objects (e.g., cars, pedestrians, traffic lights) in images, based on annotations provided in a CSV format.

Project Overview
This project utilizes a custom ObjectDetectionDataset class to load and process a dataset of images and corresponding annotations, with bounding boxes and class labels. The images are passed through convolutional layers followed by pooling layers to learn features, and the final output is passed through fully connected layers for classification and bounding box prediction.

Key Components:
Custom Dataset: The ObjectDetectionDataset class reads images and annotations, preprocesses them, and prepares them for training.
Model Architecture: The model consists of convolutional layers, followed by pooling layers, and fully connected layers for output.
Training Pipeline: The model is trained using the DataLoader to load data in batches and CrossEntropyLoss to calculate the los
3 Convolutional Layers: Extract features from images.
3 Max-Pooling Layers: Reduce the spatial dimensions of the feature maps by half after each convolutional layer.
2 Fully Connected Layers: Perform the final decision-making, mapping the learned features to both bounding box coordinates and class probabilities
