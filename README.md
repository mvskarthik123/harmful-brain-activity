# Harmful Brain Activity Classification

This project is focused on the classification of harmful brain activity using EEG data and spectrograms. The dataset consists of EEG recordings that are analyzed to detect various types of harmful brain activities such as seizures and disorders. The model is built using TensorFlow and KerasCV for deep learning-based classification.

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [License](#license)

## Overview

This project classifies brain activity into six categories: Seizure, LPD (Lateralized Paroxysmal Discharges), GPD (Generalized Paroxysmal Discharges), LRDA (Lateralized Rhythmic Discharges Abnormalities), GRDA (Generalized Rhythmic Discharges Abnormalities), and Other. It uses EEG data and spectrograms as input for training and validation.

## Tech Stack

- **Python**: 3.x
- **TensorFlow**: 2.15.0
- **Keras**: 3.3.3
- **KerasCV**: 0.9.0
- **NumPy**: 1.26.4
- **pandas**: 2.0.2
- **Matplotlib**: 3.7.1
- **OpenCV**: 4.7.0
- **Joblib**: 1.3.0
- **scikit-learn**: 1.3.0
- **TQDM**: 4.66.0
- **JAX**: 0.4.13

## Installation

To install the required dependencies, run the following commands:

```bash
pip install tensorflow==2.15.0
pip install keras==3.3.3
pip install keras_cv==0.9.0
pip install pandas matplotlib opencv-python scikit-learn tqdm joblib
```

Ensure you have Python 3.10 or a compatible version.

## Dataset

The dataset used in this project is the **Harmful Brain Activity Classification (HBAC)** dataset, which contains EEG signals and their corresponding spectrograms. The dataset is available in Kaggle and consists of labeled brain activity data. The data includes multiple categories like Seizures, LPD, and others.

### Files in Dataset:
- `train.csv`: Contains metadata for the training dataset.
- `test.csv`: Contains metadata for the test dataset.
- `train_eegs/`: Folder containing the EEG data for training.
- `train_spectrograms/`: Folder containing the spectrograms for training.
- `test_eegs/`: Folder containing the EEG data for testing.
- `test_spectrograms/`: Folder containing the spectrograms for testing.

## Usage

1. **Data Processing**: The code processes the EEG and spectrogram data and saves them as `.npy` files for faster loading.
2. **Model Training**: The model is trained using the spectrograms and their corresponding labels using deep learning techniques. The model is built using KerasCV, with augmentations for better generalization.
3. **Model Evaluation**: The model is evaluated using the validation dataset to assess its performance.

```bash
python train_model.py
```

The training process uses a predefined set of configurations in the `CFG` class. The number of epochs, batch size, learning rate, and other hyperparameters are set here.

## Model Architecture

The model is built using **KerasCV** and **TensorFlow**, using the **EfficientNetV2** architecture pretrained on ImageNet. The following key steps are involved:

- **Data Augmentation**: MixUp and RandomCutout augmentations are used to increase the diversity of the training data.
- **Normalization**: The spectrograms are normalized and standardized.
- **Model Training**: The model is trained using KLDivergence as the loss function.
- **Stratified Cross-Validation**: The dataset is split into 5 folds using **StratifiedGroupKFold** to ensure each fold has a similar class distribution.

```python
model = keras_cv.models.ImageClassifier.from_preset(
    "efficientnetv2_b2_imagenet", num_classes=6, dropout_rate=0.5
)
```
