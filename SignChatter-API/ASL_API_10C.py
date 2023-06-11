'''
Use asynchronous I/O with asyncio to allow for non-blocking I/O operations.
Use a ThreadPoolExecutor to run computationally intensive tasks, such as the keypoint prediction and model inference, in separate threads.
Use a batch size greater than 1 to improve model inference performance. Instead of predicting on a single sequence of 50 frames, predict on a batch of multiple sequences simultaneously.
Use OpenCV's VideoCapture.set() method to set the frame skip interval, instead of computing it manually.
Move the loading of the model weights to the prediction function, so that the weights are only loaded when needed.
'''

import os

import cv2
import mediapipe
import numpy as np
from fastapi import File, UploadFile, FastAPI
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense
from tensorflow.keras.models import Sequential
from fastapi.middleware.cors import CORSMiddleware


def load_model(model_path):
    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu', input_shape=(50, 258)),
        Dropout(0.2),
        LSTM(256, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(256, return_sequences=False, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights(model_path)

    return model


def predict(video_content, lstm_model):
    sequence = []
    cap = cv2.VideoCapture(video_content)

    mp_holistic = mediapipe.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        toskip = int(frame_count // 50)
        if toskip == 0:
            toskip = 1

        frame_num = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            frame_num += toskip

#            # rotate video right way up
#            (h, w) = frame.shape[:2]
#            rotpoint = (w // 2, h // 2)
#            rotmat = cv2.getRotationMatrix2D(rotpoint, 180, 1.0)
#            dim = (w, h)
#            intermediateFrame = cv2.warpAffine(frame, rotmat, dim)

            # cropping
            size = frame.shape
            finalFrame = frame[80:(size[0] - 180), 0:(size[1] - 20)]
            cv2.imwrite('fff.jpg',finalFrame)

            # keypoint prediction
            image = cv2.cvtColor(finalFrame, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False  # Image is no longer writeable
            results = holistic.process(image)  # Make prediction
            image.flags.writeable = True  # Image is now writeable
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR

            # extract and append keypoints
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                             results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
            lh = np.array([[res.x, res.y, res.z] for res in
                           results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
                21 * 3)
            rh = np.array([[res.x, res.y, res.z] for res in
                           results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
                21 * 3)
            keypoints = np.concatenate([pose, lh, rh])
            sequence.append(keypoints)

            if len(sequence) == 50:
                break

    cap.release()
    cv2.destroyAllWindows()
    sequence = np.expand_dims(sequence, axis=0)[0]
    res = lstm_model.predict(np.expand_dims(sequence, axis=0))
    os.remove(video_content)
    #    if float(np.max(res)) > float(0.9):
    #        return str(actions[np.argmax(res)])
    #    else:
    #        return 'Action Not recognized. Try Again'

    return str(actions[np.argmax(res)])


app = FastAPI()

origins = [
    "http://194.163.183.60:19470",
    "https://signchatter.netlify.app/",
    "https://signchatter.netlify.app/#/",
    "signchatter.netlify.app/#/",
    "http://localhost:5173/#/",
    "http://localhost:5173/",
    "localhost:5173/",
    ""
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = r"/root/final_year_project/Mediapipe-API-PSL/10C_50F_OG2.h5"
model = load_model(model_path)
actions = ['banana', 'bar',
           'basement', 'basketball', 'bath', 'bathroom', 'bear', 'beard', 'bed', 'bedroom']


@app.post("/")
async def predict_action(file: UploadFile = File()):
    video_path = f"/root/final_year_project/Mediapipe-API-PSL/videos2/{file.filename}"
    with open(video_path, "wb") as video:
        video.write(await file.read())
    return {'Prediction': predict(video_path, model)}
