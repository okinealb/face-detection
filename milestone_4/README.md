# Milestone 4: Real-Time Face Detection
This folder contains the implementation of a real-time face detection system that displays bounding boxes and confidence scores.


## Goals
- Optimize the trained classifier for real-time performance using live video feeds or webcam input.
- Display bounding boxes and confidence scores for detected faces in real time.
- Assess the system's real-time accuracy and computational efficiency.

## Results
- Did not reach live video performance.
- Bounding boxes and confidence scores were displayed effectively, with minimal latency.
- Optimization techniques, including downscaling and region of interest processing, ensured reliable real-time performance even on modest hardware.
- Larger training data and more flexible weighing processes were used to reduce early convergence and solve initial no convergence. 

## Instructions for Using Final_Project
- On the terminal, run the command "python cascade_trainer.py"
- It will prompt you with the options of training or testing.
- You can choose to train but be aware that this replaces the current cascade_model.pkl file placed in the folder.
- If you decide to test, place your testing images in test_images
- The results will appear in the results folder.

The folder structure is as follows:

├──Final_Project/
    ├── data/
        ├──faces/           # Positive Samples
        ├──non_faces/       # Negative Samples
        ├──results/         # Results of Testing 
        ├──test_images/     # Folder for inputting test images
    ├── cascade_model.pkl   # Pickle File with our best trained model so far
    ├── cascade_trainer.py  # Our Final Project File that includes all functions
    ├── README.md           # Provides Instructions for Running Final Project   

   