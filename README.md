# Deep-Learning-Project---Image-Classification
# Deep Learning Project README

## Overview

This project focuses on image classification using Convolutional Neural Networks (CNNs). It includes two models with different approaches to data handling and training. The dataset consists of frames from Egyptian movies, manually labeled and categorized.

## Model 1: Custom CNN

### Dataset Preparation
- **Data Creation**: Frames from movies saved in category-specific folders.
- **Data Transformation**: Images resized to (240, 360) pixels using `transforms`.
- **Data Loading**: Used `ImageFolder` and `DataLoader`.
- **Data Split**: Training (70%), validation (20%), and testing (10%).

### CNN Layers
- **Custom Layers**: `InputLayer`, `ConvLayer`, `PoolingLayer`, `FlatteningLayer`, `DownsamplingLayer`.
- **Functionality**: Convolution, pooling, flattening, and downsampling operations.

### Training & Testing
- **Forward Propagation**: Applied filters and layers sequentially.
- **K-means Clustering**: Feature extraction and training.
- **Classification**: Predicted labels with trained K-means clusters.

### Results
- **Training Accuracy**: 85%
- **Validation Accuracy**: 80%
- **Test Accuracy**: 78%

## Model 2: TensorFlow CNN

### Setup
- **Environment**: TensorFlow with GPU acceleration.
- **Libraries**: `numpy`, `sklearn`, `matplotlib`, `tensorflow.keras`.

### CNN Architecture
- **Layers**: Sequential model with convolutional, max pooling, and dense layers.
- **Activations**: ReLU, sigmoid, and softmax.

### Training Process
- **K-fold Cross-validation**: `KFold` from sklearn, k=4.
- **Data Augmentation**: `ImageDataGenerator`.
- **Evaluation**: Accuracy and confusion matrix.

### Results
- **Average Accuracy**: 87.85%
- **Confusion Matrix**: Shows detailed per-class performance.

### Inference
- **Random Testing**: Selected a random test image and displayed prediction results.

## Data Link
[Dataset on Google Drive](https://drive.google.com/drive/folders/1OMhSRm7ZCwbZuEc5KoPSdgTVvP0HcioJ?usp=drive_link)

## Conclusion
The project demonstrates the implementation of two CNN models for image classification, detailing steps for dataset preparation, model building, training, and evaluation with specific results.
