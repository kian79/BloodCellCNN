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

**Note**: Ensure you have the `training_set.npz` file in the project directory.

## Installation

### Prerequisites

- Python 3.8 or higher
- [Google Colab](https://colab.research.google.com/) (optional, for running the notebook)

### Clone the Repository

```bash
git clone https://github.com/yourusername/BloodCellClassifier.git
cd BloodCellClassifier
