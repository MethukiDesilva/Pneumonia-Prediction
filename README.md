# Chest X-Ray Pneumonia Classification

A convolutional neural network (CNN) that classifies chest X-ray images as either Normal or Pneumonia.

## Overview

This project trains a CNN from scratch to detect pneumonia from chest X-ray images. It covers the full pipeline: data loading and preprocessing, handling class imbalance, model training with early stopping, evaluation, and inference on new images.

Dataset
Source: Kaggle — Chest X-Ray Images (Pneumonia) https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images?select=chest_xray
Structure: Pre-split into train/ and test/ folders, each containing NORMAL and PNEUMONIA subfolders of grayscale JPEG images.
Class balance: The training set is imbalanced at roughly a 3:1 ratio of Pneumonia to Normal images.
xray/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
## Approach
Preprocessing — Images are loaded as grayscale (chest X-rays carry no real color information), resized to 150x150, and normalized to a 0–1 pixel range.
Splitting — The training folder is further split into train/validation sets (85/15, stratified). The test folder is kept fully separate and untouched until final evaluation.
Class weighting — Because the dataset is imbalanced, class weights (sklearn.utils.class_weight.compute_class_weight) are applied during training so the model isn't biased toward the majority class.
Model — A CNN with three convolutional blocks (32 → 64 → 128 filters, each followed by max pooling), a dense layer with dropout, and a softmax output over the two classes.
Training — Uses EarlyStopping (monitoring validation loss, patience of 3 epochs) to prevent overfitting and automatically restore the best-performing weights.
Evaluation — Accuracy, a confusion matrix, and per-class precision/recall/F1 are reported, since overall accuracy alone can be misleading on an imbalanced dataset.

## Requirements
tensorflow
opencv-python
numpy
matplotlib
seaborn
scikit-learn


pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn

By Methuki De Silva**
