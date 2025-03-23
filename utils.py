import cv2
import mediapipe as mp
import joblib
import numpy as np
import torch
import torch.nn as nn

#Helper function to get face landmarks
def get_landmarks(img,static_mode=True):
    if img is None:
        return None 
    mp_face_mesh=mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=static_mode,max_num_faces=1,min_detection_confidence=0.5,refine_landmarks=True) as face_mesh:
        results=face_mesh.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks=[]
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    landmarks.extend([landmark.x,landmark.y,landmark.z])
            return landmarks
        else:
            return None

#Helper function which loads scaler,one hot encoder and the ANN model        
def load_models(scaler_path,one_hot_encoder_path,model_path):
    scaler = joblib.load(scaler_path)  
    one_hot_encoder = joblib.load(one_hot_encoder_path)
    class ANN(nn.Module):
        def __init__(self, input_size=1441, hidden_size1=128, hidden_size2=64, hidden_size3=32, output_size=4):
            super(ANN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.bn1 = nn.BatchNorm1d(hidden_size1)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.3)

            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.bn2 = nn.BatchNorm1d(hidden_size2)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.3)

            self.fc3 = nn.Linear(hidden_size2, hidden_size3)
            self.bn3 = nn.BatchNorm1d(hidden_size3)
            self.relu3 = nn.ReLU()

            self.fc4 = nn.Linear(hidden_size3, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.dropout1(x)

            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.dropout2(x)

            x = self.fc3(x)
            x = self.bn3(x)
            x = self.relu3(x)

            x = self.fc4(x)
            return x
    model = ANN(input_size=1441)
    model.load_state_dict(torch.load(model_path))
    return scaler,one_hot_encoder,model

#Helper function which gives the prediction scores
def predict_class(img,scaler,one_hot_encoder,model):
    landmarks=get_landmarks(img)
    if landmarks is None or landmarks==[]:
        return None,None
    landmarks+=[landmarks[291*3]-landmarks[61*3]]+[landmarks[14*3+1]-landmarks[13*3+1]]+[landmarks[105*3+1]-landmarks[159*3+1]]+[landmarks[334*3+1]-landmarks[286*3+1]]+[landmarks[300*3]-landmarks[70*3+1]]+[landmarks[145*3+1]-landmarks[159*3+1]]+[landmarks[374*3+1]-landmarks[386*3+1]]
    X = np.array(landmarks).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
    probabilities = torch.softmax(output, dim=1).numpy()[0]
    predicted_class = np.argmax(probabilities)
    predicted_one_hot = np.zeros((1, 4))  
    predicted_one_hot[0, predicted_class] = 1
    original_label = one_hot_encoder.inverse_transform(predicted_one_hot)[0][0]
    return original_label,probabilities