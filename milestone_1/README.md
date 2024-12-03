# Milestone 1: Preliminary Face Detection
This folder contains the implementation of face detection using OpenCV's pretrained Haar Cascade Classifier. The goal of this milestone was to validate the feasability of the project.

## Goals
- Implement and test OpenCV's pretrained Haar Cascade Classifier for face detection.
- Detect faces in static images and visualize the results by drawing bounding boxes around detected faces.
- Evaluate the classifier’s performance on the BioID Face Database to understand the challenges in face detection under varying conditions.

## Results
- The Haar Cascade Classifier successfully detected faces in a majority of the images from the BioID Face Database.
- Bounding boxes were accurately drawn around detected faces, with occasional false positives and missed detections in challenging cases (e.g., poor lighting or partial occlusion).
- The results confirmed that OpenCV’s Haar Cascade provides a solid foundation for further customization and improvement in subsequent milestones.

## Acknowledgements
- OpenCV Library: For providing pretrained Haar Cascade Classifiers, and the tools and resources needed for efficient real-time processing.
- Paul Viola and Michael Jones: For their foundational work on the Haar Cascade, integral image methods, and AdaBoost.
- BioID Face Database: For the dataset used throughout this project.