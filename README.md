Emotion_Recognition
A model to detect and recognize facial expressions and emotions in a real time video/webcam input environment
## Overview
This project detects and recognizes common human emotions such as happy, sad, neutral, surprised, and angry using OpenCV, PyTorch, and MediaPipe.

## File Structure
|-csv_files <br>
|  &nbsp;&nbsp;|-data.csv (original data , shared the link in acknowledgements)<br>
|  &nbsp;&nbsp;|-landmarks.csv (Repo doesn't contain this file due to size limit restrictions but the code for building this file is available in data_preparation.ipynb)<br>
|-models<br>
|  &nbsp;&nbsp;|-emotion_model_weights.pth (classification model)<br>
|  &nbsp;&nbsp;|-one_hot_encoder.pkl (one_hot_encoder)<br>
|  &nbsp;&nbsp;|-scaler.pkl (scaler)<br>
|-LICENSE<br>
|-README.md <br>
|-classifier.ipynb (classification mdoel)<br>
|-data_preparation.ipynb (data preprocessing and preparation)<br>
|-utils.py (contains helper functions)<br>
|-webcam_deployment.py (deployment file)<br>

## Usage
Run the webcam deployment script to start emotion recognition:<br>
bashCopypython webcam_deployment.py<br>
## Features
Real-time facial emotion recognition<br>
Detection of 5 common emotions: happy, sad, neutral, surprised, angry<br>
Facial landmark detection using MediaPipe<br>
ML classification model built with PyTorch<br>

## Requirements

Python 3.9<br>
OpenCV<br>
PyTorch<br>
MediaPipe<br>
NumPy<br>
Pandas<br>
scikit-learn<br>

## Acknowledgements and copyrights
Original dataset: https://www.kaggle.com/datasets/karthickmcw/emotion-recognition-dataset <br>
MediaPipe for facial landmark detection<br>
