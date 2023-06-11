import math
import multiprocessing
import os
import shutil
import numpy as np
import mediapipe as mp
import cv2
import time



# function to draw landmarks on frames for visualization purposes
def draw_styled_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic  # Holistic model# Drawing utilities
    # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          )
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


# smaller function to apply mediapipe detection on passed frame
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR

    return image, results


# smaller function to extract keypoints and return as an np array
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, lh, rh])


# function to extract keypoints for each frame and store as an npy file
def action_division(actions):
    save_to = r"D:\ASL\NewKeyPoints"
    path = r"D:\ASL\NewDataSet"
    mp_holistic = mp.solutions.holistic  # Holistic model

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:  # Loop through actions
            print(action)
            videos = os.listdir(os.path.join(path, action))

            for video in videos:
                augs = os.listdir(os.path.join(path, action, video))
                print(video)

                for aug in augs:
                    print(aug)
                    frames = os.listdir(os.path.join(path, action, video, aug))

                    for frame in frames:
                        npy_path = os.path.join(save_to, str(action), str(video), str(aug), str(frame))  # npy file name
                        if os.path.exists(npy_path):
                            os.remove(npy_path)
                            # continue
                        img_path = os.path.join(path, action, video, aug, frame)
                        img = cv2.imread(img_path)
                        image, results = mediapipe_detection(img, holistic)
                        keypoints = extract_keypoints(results)
                        np.save(npy_path, keypoints)


# To check how many actions have been completed
def check(actions):
    complete = []
    var = False
    path = r"D:\ASL\NewKeyPoints"
    for action in actions:
        print(action)
        var = False
        videos = os.listdir(os.path.join(path, action))
        for video in videos:
            augs = os.listdir(os.path.join(path, action, video))
            for aug in augs:
                frames = os.listdir(os.path.join(path, action, video, aug))
                if len(frames) < 50:
                    var = True
                    break
        if var == False:
            complete.append(action)
    print(len(complete))
    print(complete)


def get_arrays():
    completed_list = ['about', 'accident', 'africa', 'afternoon', 'again', 'all', 'always', 'animal', 'any', 'apple',
                      'approve', 'argue', 'arrive', 'aunt', 'baby', 'back', 'bake', 'balance', 'bald', 'ball', 'banana',
                      'bar', 'basement', 'basketball', 'bath', 'bathroom', 'bear', 'beard', 'bed', 'bedroom', 'before',
                      'better', 'bicycle', 'bird', 'birthday', 'bitter', 'black', 'blue', 'book', 'both', 'bowl',
                      'bowling', 'box', 'boy', 'boyfriend', 'bring', 'brother', 'brown', 'business', 'but', 'buy',
                      'can', 'candy', 'cards', 'cat', 'catch', 'center', 'cereal', 'chair', 'change', 'check', 'cheese',
                      'chicken', 'children', 'christmas', 'church', 'city', 'class', 'classroom', 'clock', 'clothes',
                      'coffee', 'cold', 'college', 'color', 'computer', 'cook', 'cookie', 'cool', 'copy', 'corn',
                      'cough', 'country', 'cousin', 'cow', 'crash', 'crazy', 'cute', 'dance', 'dark', 'day', 'deaf',
                      'decide', 'dentist', 'dictionary', 'different', 'dirty', 'discuss', 'doctor', 'dog', 'doll',
                      'door', 'draw', 'drink', 'easy', 'eat', 'elevator', 'enjoy', 'enter', 'environment', 'exercise',
                      'experience', 'face', 'family', 'far', 'fat', 'feel', 'fine', 'finish', 'first', 'fish',
                      'fishing', 'food', 'football', 'forget', 'friend', 'friendly', 'from', 'full', 'future', 'game',
                      'give', 'glasses', 'go', 'graduate', 'greece', 'green', 'hair', 'halloween', 'happy', 'hat',
                      'have', 'headache', 'hearing', 'help', 'here', 'home', 'hospital', 'hot', 'house', 'how',
                      'husband', 'interest', 'internet', 'investigate', 'jacket', 'jump', 'kiss', 'knife', 'know',
                      'language', 'last', 'last year', 'later', 'law', 'learn', 'letter', 'library', 'like', 'list',
                      'lose', 'lunch', 'magazine', 'man', 'many', 'match', 'mean', 'meat', 'medicine', 'meet',
                      'meeting', 'money', 'moon', 'more', 'most', 'mother', 'movie', 'music', 'name', 'need',
                      'neighbor', 'nephew', 'never', 'nice', 'niece', 'no', 'none', 'noon', 'north', 'not', 'now',
                      'nurse', 'off', 'ok', 'old', 'opinion', 'orange', 'order', 'paint', 'paper', 'pencil', 'people',
                      'perspective', 'phone', 'pink', 'pizza', 'plan', 'play', 'please', 'police', 'potato', 'practice',
                      'president', 'pull', 'purple', 'rabbit', 'rain', 'read', 'red', 'remember', 'restaurant', 'ride',
                      'run', 'sad', 'school', 'science', 'secretary', 'sentence', 'share', 'shirt', 'shoes', 'shop',
                      'sick', 'sign', 'since', 'sister', 'sleep', 'slow', 'small', 'smile', 'snow', 'some', 'son',
                      'sorry', 'south', 'star', 'straight', 'strange', 'struggle', 'student', 'study', 'sunday',
                      'suspect', 'sweetheart', 'table', 'tall', 'tea', 'teacher', 'tent', 'test', 'thank you',
                      'thanksgiving', 'thin', 'think', 'thirsty', 'thursday', 'tiger', 'time', 'tired', 'toast',
                      'together', 'toilet', 'tomorrow', 'traffic', 'travel', 'tree', 'truck', 'uncle', 'use', 'visit',
                      'wait', 'walk', 'want', 'war', 'water', 'weak', 'wednesday', 'week', 'what', 'when', 'where',
                      'which', 'white', 'who', 'why', 'wife', 'win', 'wind', 'window', 'with', 'woman', 'work', 'world',
                      'worry', 'write', 'wrong', 'year', 'yellow', 'yes', 'yesterday', 'you']

    # completed_list = set(completed_list)
    # actions = os.listdir(r'D:\ASL\NewDataSet')
    # actions = set(actions)
    # actions = actions - completed_list
    # actions = list(actions)
    # action_division(actions)


    # length = len(actions) // 6
    # first = actions[: length]
    # second = actions[length: length * 2]
    # third = actions[length * 2: length * 3]
    # fourth = actions[length * 3: length * 4]
    # fifth = actions[length * 4: length * 5]
    # sixth = actions[length * 5:]

    # p1 = multiprocessing.Process(target=action_division, args=(first,))
    # p2 = multiprocessing.Process(target=action_division, args=(second,))
    # p3 = multiprocessing.Process(target=action_division, args=(third,))
    # p4 = multiprocessing.Process(target=action_division, args=(fourth,))
    # p5 = multiprocessing.Process(target=action_division, args=(fifth,))
    # p6 = multiprocessing.Process(target=action_division, args=(sixth,))

    # p1 = multiprocessing.Process(target=check, args=(first,))
    # p2 = multiprocessing.Process(target=check, args=(second,))
    # p3 = multiprocessing.Process(target=check, args=(third,))
    # p4 = multiprocessing.Process(target=check, args=(fourth,))
    # p5 = multiprocessing.Process(target=check, args=(fifth,))
    # p6 = multiprocessing.Process(target=check, args=(sixth,))

    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()




if __name__ == '__main__':
    pass
    # action_division(actions)
    # get_arrays()
    # actions = os.listdir(r'D:\ASL\NewDataSet')
    # check(actions)
