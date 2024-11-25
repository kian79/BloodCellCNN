# BloodCellCNN

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.11.0-blue.svg)

**Deep Learning Model for Classifying 8 Blood Cell Types**

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project focuses on the classification of blood cell images into **8 distinct classes** using advanced deep learning techniques. Leveraging state-of-the-art convolutional neural network architectures, the goal is to achieve high accuracy in distinguishing between different types of blood cells, which is crucial for medical diagnostics and research.

## Features

- **Data Preprocessing**: Includes outlier removal and data normalization.
- **Data Augmentation**: Implements various augmentation techniques to enhance model robustness.
- **Model Ensemble**: Combines multiple models (EfficientNetB3, MobileNetV2, ResNet50) to improve prediction accuracy.
- **Performance Metrics**: Evaluates models using accuracy, precision, recall, and confusion matrix.
- **Visualization**: Provides comprehensive visualizations of data distribution and model performance.

## Dataset

The dataset consists of blood cell images categorized into 8 classes. Each image is preprocessed and stored in a NumPy `.npz` file containing:

- `images`: Array of image data.
- `labels`: Corresponding class labels.

Here is a photo of different classes in the dataset.
![Images](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

## Model Architecture

The project employs three different pre-trained convolutional neural network architectures as base models:

- EfficientNetB3
- MobileNetV2
- ResNet50
Each model is customized with:

- Batch Normalization
- Global Average Pooling
- Dropout Layers
- Dense Output Layer with Softmax Activation

Data Augmentation Layers:

- Random Flips, Rotations, Zooms, Contrast adjustments, Cropping, Resizing
- Custom layers for Gaussian Noise, Hue and Saturation adjustments.


## Training

### Data Preparation
- Outlier Removal: Eliminates specific outlier images from the dataset.
- Train-Validation-Test Split: 80% training, 10% validation, 10% testing with stratification.
- Sample Weighting: Balances class weights to handle class imbalance.

### Augmentation Techniques
- AugMix
- RandAugment
- CutMix
- MixUp
- Channel Shuffle
- Random Color Degeneration

### Training Configuration
- Epochs: 100
- Batch Size: 32
- Optimizer: Lion optimizer with a learning rate of 1e-4
- Loss Function: Categorical Crossentropy
- Callbacks: Early Stopping and Learning Rate Scheduler


### Training Process
The models are trained individually with their respective configurations. After training, each model's weights are saved for ensemble prediction.

## Results

### Model Performance
Each model's accuracy on the test set is as belows. 

- EfficientNetB3: Up to 96.5%
- MobileNetV2: Up to 95.2%
- ResNet50: Up to 94.5%

### Ensemble Performance
Combining predictions from all models yields improved accuracy:

- Test Accuracy: 97.74
- Precision: 97.97
- Recall: 97.74

### Confusion Matrix
A confusion matrix visualizes the performance across different classes, highlighting areas of high accuracy and potential misclassifications.

![Confusion Matrix](https://github.com/[kian79]/BloodCellCNN/blob/[branch]/image.jpg?raw=true)

### Visualization
The project includes comprehensive visualizations to understand data distribution and model performance:

- Data Distribution: Shows the number of samples per class.
- Sample Images: Displays sample images from each class.
- Training Curves: Plots of loss and accuracy over epochs.
- Confusion Matrix: Detailed performance across classes.
