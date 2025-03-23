import cv2
from utils import load_models, predict_class

# Load the scaler,one hot encoder and model
scaler, one_hot_encoder, model = load_models(
    r"D:\emotion_dataset\models\scaler.pkl", #scaler path
    r"D:\emotion_dataset\models\one_hot_encoder.pkl", #one hot encoder path
    r"D:\emotion_dataset\models\emotion_model_weights.pth" #model path
)

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    result,probs = predict_class(frame, scaler, one_hot_encoder, model)
    
    predicted_emotion = result
    if result:
        #Fine tune the model decision boundaries according to your camera specs
        if probs[0]>0.009 and probs[3]<0.33:
            cv2.putText(frame, "Angry", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif probs[3]>0.22:
            cv2.putText(frame, "Surprised", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif (probs[1]-probs[2])<0.10 and (probs[1]-probs[2])>0:
            cv2.putText(frame, "Neutral", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif probs[1]>0.50:
            cv2.putText(frame, "Happy", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Sad", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    else:
        cv2.putText(frame, "No face detected" , (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()