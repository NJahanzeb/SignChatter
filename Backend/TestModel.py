import math
import os
import numpy as np
import mediapipe as mp
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense


# function to draw landmarks on extracted frames of input video
def draw_styled_landmarks(image, results):
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


# function to build model and load weights
def loadModel():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(50, 258)))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=False, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights(r'D:\Jahanzeb\PycharmProjects\wlasl\10C_50F_OG2.h5')
    return model


# words recognizable by 10C_50F_OG2.h5 model
#'banana', 'bar', 'basement', 'basketball', 'bath', 'bathroom', 'bear', 'beard', 'bed', 'bedroom'


# words recognizable by 30C_50F_128B_5T_sorted.h5 model
# 'about', 'accident', 'africa', 'afternoon', 'again', 'all', 'always', 'animal', 'any', 'apple',
# 'approve', 'argue', 'arrive', 'aunt', 'baby', 'back', 'bake', 'balance', 'bald', 'ball', 'banana', 'bar',
# 'basement', 'basketball', 'bath', 'bathroom', 'bear', 'beard', 'bed', 'bedroom'


def prediction(video_path, lstm_model):
    actions = ['about', 'accident', 'africa', 'afternoon', 'again', 'all', 'always', 'animal', 'any', 'apple',
               'approve', 'argue', 'arrive', 'aunt', 'baby', 'back', 'bake', 'balance', 'bald', 'ball', 'banana', 'bar',
               'basement', 'basketball', 'bath', 'bathroom', 'bear', 'beard', 'bed', 'bedroom', 'before', 'better',
               'bicycle', 'bird', 'birthday', 'bitter', 'black', 'blue', 'book', 'both', 'bowl', 'bowling', 'box',
               'boy', 'boyfriend', 'bring', 'brother', 'brown', 'business', 'but', 'buy', 'can', 'candy', 'car',
               'cards', 'cat', 'catch', 'center', 'cereal', 'chair', 'change', 'check', 'cheese', 'chicken', 'children',
               'christmas', 'church', 'city', 'class', 'classroom', 'clock', 'clothes', 'coffee', 'cold', 'college',
               'color', 'computer', 'cook', 'cookie', 'cool', 'copy', 'corn', 'cough', 'country', 'cousin', 'cow',
               'crash', 'crazy', 'cute', 'dance', 'dark', 'day', 'deaf', 'decide', 'dentist', 'dictionary', 'different',
               'dirty', 'discuss', 'doctor', 'dog', 'doll', 'door', 'draw', 'drink', 'easy', 'eat', 'elevator', 'enjoy',
               'enter', 'environment', 'exercise', 'experience', 'face', 'family', 'far', 'fat', 'feel', 'fine',
               'finish', 'first', 'fish', 'fishing', 'food', 'football', 'forget', 'friend', 'friendly', 'from', 'full',
               'future', 'game', 'give', 'glasses', 'go', 'graduate', 'greece', 'green', 'hair', 'halloween', 'happy',
               'hat', 'have', 'headache', 'hearing', 'help', 'here', 'home', 'hospital', 'hot', 'house', 'how',
               'husband', 'interest', 'internet', 'investigate', 'jacket', 'jump', 'kiss', 'knife', 'know', 'language',
               'last', 'last year', 'later', 'law', 'learn', 'letter', 'library', 'like', 'list', 'lose', 'lunch',
               'magazine', 'man', 'many', 'match', 'mean', 'meat', 'medicine', 'meet', 'meeting', 'money', 'moon',
               'more', 'most', 'mother', 'movie', 'music', 'name', 'need', 'neighbor', 'nephew', 'never', 'nice',
               'niece', 'no', 'none', 'noon', 'north', 'not', 'now', 'nurse', 'off', 'ok', 'old', 'opinion', 'orange',
               'order', 'paint', 'paper', 'pencil', 'people', 'perspective', 'phone', 'pink', 'pizza', 'plan', 'play',
               'please', 'police', 'potato', 'practice', 'president', 'pull', 'purple', 'rabbit', 'rain', 'read', 'red',
               'remember', 'restaurant', 'ride', 'run', 'sad', 'school', 'science', 'secretary', 'sentence', 'share',
               'shirt', 'shoes', 'shop', 'sick', 'sign', 'since', 'sister', 'sleep', 'slow', 'small', 'smile', 'snow',
               'some', 'son', 'sorry', 'south', 'star', 'straight', 'strange', 'struggle', 'student', 'study', 'sunday',
               'suspect', 'sweetheart', 'table', 'tall', 'tea', 'teacher', 'tent', 'test', 'thank you', 'thanksgiving',
               'thin', 'think', 'thirsty', 'thursday', 'tiger', 'time', 'tired', 'toast', 'together', 'toilet',
               'tomorrow', 'traffic', 'travel', 'tree', 'truck', 'uncle', 'use', 'visit', 'wait', 'walk', 'want', 'war',
               'water', 'weak', 'wednesday', 'week', 'what', 'when', 'where', 'which', 'white', 'who', 'why', 'wife',
               'win', 'wind', 'window', 'with', 'woman', 'work', 'world', 'worry', 'write', 'wrong', 'year', 'yellow',
               'yes', 'yesterday', 'you']

    sequence = []
    cap = cv2.VideoCapture(video_path)  # video path

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        toskip = int(frame_count // 50)
        if toskip == 0:
            toskip = 1

        frame_num = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            frame_num = frame_num + toskip

            # rotate video right way up (auto-rotate in OpenCV 4.6+)
            # (h, w) = frame.shape[:2]
            # rotpoint = (w // 2, h // 2)
            # rotmat = cv2.getRotationMatrix2D(rotpoint, 180, 1.0)
            # dim = (w, h)
            # intermediateFrame = cv2.warpAffine(frame, rotmat, dim)
            # intermediateFrame = frame

            # cropping
            size = frame.shape
            finalFrame = frame[50:(size[0] - 170), 0:(size[1] - 20)]

            # keypoint prediction
            image = cv2.cvtColor(finalFrame, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False  # Image is no longer writeable
            results = holistic.process(image)  # Make prediction
            image.flags.writeable = True  # Image is now writeable
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR

            draw_styled_landmarks(image, results)
            cv2.imshow('Frame', image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

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
                cap.release()
                break

    cap.release()
    cv2.destroyAllWindows()
    sequence = np.expand_dims(sequence, axis=0)[0]

    res = lstm_model.predict(np.expand_dims(sequence, axis=0))

    print(actions[np.argmax(res)])


lstm_model = loadModel()
vids = os.listdir(r'D:\Jahanzeb\PycharmProjects\wlasl\3')

for vid in vids:
    print('\n', vid)
    prediction(f'D:/Jahanzeb/PycharmProjects/wlasl/3/{vid}', lstm_model)


