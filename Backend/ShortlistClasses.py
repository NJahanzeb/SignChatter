import json
import math
import multiprocessing
import os
import cv2
import shutil
import concurrent.futures


# Code to find videos containing at least 50 frames

video_dataset = r'D:\WLASL2000'
files = os.listdir(video_dataset)
eligible_videos = []

for file in files:
    cap = cv2.VideoCapture(video_dataset + '/' + file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count >= 50:
        print(file,frame_count)
        eligible_videos.append(file)

# Code to find eligible classes/words and their eligible videos
# Modified to also make directories for eligible words/classes and their eligible videos
#
# 10 vids threshold = 322 classes
# 11 vids threshold = 225 classes
# 12 vids threshold = 153 classes

json_file = open(r'D:\ASL\WLASL_v0.3.json')
json_data = json.load(json_file)
eligible_words = []
threshold = 10
images_dataset = r"D:\ASL\NewKeyPoints"
# directory structure of frames and keypoints is the same so just re-run this code for both but use different paths

for dict_instance in json_data:
    count = 0
    videos = dict_instance['instances']
    videos_to_add = []

    for video in videos:
        id = video['video_id']
        id = id + '.mp4'
        if id in eligible_videos:
            videos_to_add.append(id)
            count = count + 1

    if count >= threshold:
        word = dict_instance['gloss']
        eligible_words.append(word)
        os.mkdir(os.path.join(images_dataset, word))

        videos_to_add = videos_to_add[:threshold]
        for video in videos_to_add:
            os.mkdir(os.path.join(images_dataset, word, video))
            for i in range(1, 56):
                print(i)
                os.mkdir(os.path.join(images_dataset, word, video, str(i)))

print(len(eligible_words))



# function to delete the directories
# def cleanup(actions, images_dataset):
#     for action in actions:
#         videos = os.listdir(os.path.join(images_dataset, action))
#         print(action)
#
#         for video in videos:
#             augs = os.listdir(os.path.join(images_dataset, action, video))
#
#             for aug in augs:
#                 length = os.listdir(os.path.join(images_dataset, action, video, aug))
#                 shutil.rmtree(os.path.join(images_dataset, action, video, aug))

