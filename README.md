# 😃 Emotion Recognition

A real-time facial expression recognition system using **OpenCV**, **PyTorch**, and **MediaPipe**.

---

## 📌 Overview

This project detects and recognizes common human emotions such as:

- 😊 Happy  
- 😢 Sad  
- 😐 Neutral  
- 😲 Surprised  
- 😠 Angry  

It uses facial landmark detection and a trained PyTorch classifier to perform real-time inference through a webcam.

---

## 📁 File Structure

emotion_recognition/ <br>
├── csv_files/<br>
│   ├── data.csv              # Original raw dataset <br>
│   └── landmarks.csv         # Preprocessed facial landmarks (not included due to size)<br>
│<br>
├── models/<br>
│   ├── emotion_model_weights.pth   # Trained PyTorch model<br>
│   ├── one_hot_encoder.pkl         # Saved encoder<br>
│   └── scaler.pkl                  # Scaler for normalization<br>
│<br>
├── classifier.ipynb         # Emotion classification model training<br>
├── data_preparation.ipynb   # Data preprocessing and landmark extraction<br>
├── webcam_deployment.py     # Real-time webcam emotion detection<br>
├── utils.py                 # Helper functions<br>
├── LICENSE<br>
└── README.md<br>


---

## 🚀 Usage

To run the real-time webcam emotion recognition:
python webcam_deployment.py

## 💡 Features
- 🕵️‍♂️ Real-time emotion detection from webcam  
- 🎭 Recognizes 5 basic human emotions  
- 🧠 Lightweight ML model built using PyTorch  
- 👁️ Facial landmark detection via MediaPipe  
- 📉 Preprocessing pipeline with normalization and one-hot encoding  

---

## 🧰 Requirements

Make sure you have Python 3.9 and install the required packages using this command:

pip install opencv-python mediapipe torch numpy pandas scikit-learn

---

## 📦 Dataset

Original dataset used for training:

📂 [Kaggle Emotion Recognition Dataset](https://www.kaggle.com/datasets/karthickmcw/emotion-recognition-dataset)

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev) for efficient and accurate facial landmark detection  
- [Karthick MCW's Kaggle dataset](https://www.kaggle.com/datasets/karthickmcw/emotion-recognition-dataset)


