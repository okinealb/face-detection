# Face Detection Project - CSC 262 Independent Project
CSC 262 Project

## Authors
Albert-Kenneth Okine and Shibam Mukhopadhyay

## Overview
This project focuses on implementing a face detection system designed to detect faces in images and videos using Haar Cascade classifiers. Beginning with OpenCV's pretrained facial detection classifiers, we aim to develeop custom Haar Cascade filters for optimized face detection. The system will eventually output bounding boxes around detected faces, complete with confidence scores.

## Key Milestones
1. **Milestone 1:** Implement preliminary face detection using OpenCV's pretrained Haar Cascade classifier
2. **Milestone 2:** Implement image integrals and extract Haar features from images
3. **Milestone 3:** Train a face detection classifier using AdaBoost based on the Haar features
4. **Milestone 4:** Implement real-time face detection and output confidence scores on live video feeds

## Datasets
We will use the BioID Face Database, Labeled Faces in the Wild Dataset and Caltech 101 Dataset. These will include training and testing data for faces both positive and negative samples.

## Evaluation
The project's success will be evaluated based on:
- Comparison with OpenCV's pretrained Haar Cascade Classifier.
- Performance on real-time face detection (FPS and accuracy metrics)
- Robustness under varying lighting, face sizes, poses, and number of faces

## Repository Structure
```
face-detection/
│   README.md
├── face_database/      # Frontal face images and their eye positions
├── milestone_1/        # Preliminary facial detection
├── milestone_2/        # Image integrals and features 
├── milestone_3/        # Face detection classifier training
├── milestone_4/        # Optimization of Face Detection
    ├──Final_Project/   # Our Final trained model
    ├── README.md       # Provides Instructions for Running Final Project
├── reports/            # Detailed reports on each milestone