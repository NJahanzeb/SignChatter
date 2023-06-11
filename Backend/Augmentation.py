import math
import multiprocessing
import os
import shutil
import threading
import time
import numpy as np
import cv2


# function to extract and store 50 frames from shortlisted videos of shortlisted classes
def getOriginalFrames(actions, images_dataset, dataset_videos):
    all_videos = os.listdir(dataset_videos)  # list of actual videos
    categories = actions  # list of eligible words/classes

    for name in categories:
        print("getOriginalFrames function now in class: ", name)
        videos = os.listdir(os.path.join(images_dataset, name))

        for video in videos:
            if video in all_videos:
                cap = cv2.VideoCapture(dataset_videos + '/' + video)
                to_capture = 50
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                toskip = math.floor(frame_count / to_capture)
                if toskip == 0:
                    toskip = 1

                frame_num = 0
                count = 1
                while (cap.isOpened()):
                    if len(os.listdir(os.path.join(images_dataset, name, video, '1'))) == 50:
                        break
                    else:
                        ret, frame = cap.read()
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        frame_num = frame_num + toskip
                        if ret == True:

                            # Display the resulting frame
                            # cv2.imshow('Frame', frame)

                            os.chdir(os.path.join(images_dataset, name, video, '1'))
                            if os.path.exists((str(count) + '.jpg')):
                                os.remove((str(count) + '.jpg'))
                            cv2.imwrite((str(count) + '.jpg'),
                                        frame)  # must change directory first otherwise imwrite won't work
                            count = count + 1

                            # Press Q on keyboard to  exit
                            # if cv2.waitKey(25) & 0xFF == ord('q'):
                            #     break

                        else:
                            cap.release()
                            break
                cap.release()
                cv2.destroyAllWindows()


# data augmentation function to crop extracted frames
def Crop_Videos(images_dataset):
    categories = os.listdir(images_dataset)  # list of eligible words/classes
    for action in categories:
        print("Crop_Videos function now in class: ", action)
        videos = os.listdir(os.path.join(images_dataset, action))

        for video in videos:
            frames = os.listdir(os.path.join(images_dataset, action, video, '1'))

            for frame in frames:
                try:
                    img = cv2.imread(os.path.join(images_dataset, action, video, '1', frame))
                    size = img.shape
                    newFrame = img[100:(size[0] - 30), 50:(size[1] - 50)]
                    # cv2.imshow('Frame', newFrame)
                    os.chdir(os.path.join(images_dataset, action, video, '2'))
                    if os.path.exists(frame):
                        os.remove(frame)
                    cv2.imwrite(frame, newFrame)
                    # if cv2.waitKey(25) & 0xFF == ord('q'):
                    #     break
                except AttributeError:
                    continue


# data augmentation function to skew/rotate extracted frames
def SkewFrames(angle, images_dataset):
    folder = 0
    if angle == -5:
        folder = 3
    elif angle == 5:
        folder = 4
    elif angle == -10:
        folder = 5
    elif angle == 10:
        folder = 6
    elif angle == -15:
        folder = 7
    elif angle == 15:
        folder = 8
    elif angle == -20:
        folder = 9
    elif angle == 20:
        folder = 10
    folder = str(folder)

    categories = os.listdir(images_dataset)  # list of eligible words/classes

    for action in categories:
        print("SkewFrames function now in class: ", action)
        videos = os.listdir(os.path.join(images_dataset, action))

        for video in videos:
            frames = os.listdir(os.path.join(images_dataset, action, video, '1'))

            for frame in frames:
                try:
                    img = cv2.imread(
                        os.path.join(images_dataset, action, video, '1', frame))  # Display the resulting frame
                    (h, w) = img.shape[:2]
                    rotpoint = (w // 2, h // 2)
                    rotmat = cv2.getRotationMatrix2D(rotpoint, angle, 1.0)
                    dim = (w, h)
                    newFrame = cv2.warpAffine(img, rotmat, dim)

                    os.chdir(os.path.join(images_dataset, action, video, folder))
                    if os.path.exists(frame):
                        os.remove(frame)
                    cv2.imwrite(frame, newFrame)

                    # cv2.imshow('Frame', newFrame)
                    # if cv2.waitKey(25) & 0xFF == ord('q'):
                    #     break
                except AttributeError:
                    print('AttributeError in SkewFrames func')
                    continue
                    # Display the resulting frame


# data augmentation function to extract and rescale frames
def ResizeFrames(size, images_dataset, dataset_videos):
    folder = 0
    if size == 50:
        folder = 11
    elif size == 150:
        folder = 12
    elif size == 200:
        folder = 13
    elif size == 250:
        folder = 14
    if size == 300:
        folder = 15
    folder = str(folder)

    all_videos = os.listdir(dataset_videos)  # list of actual videos
    categories = os.listdir(images_dataset)  # list of categories

    for name in categories:
        print("ResizeFrames function now in class: ", name)
        selected_videos = os.listdir(os.path.join(images_dataset, name))

        for video in selected_videos:
            if video in all_videos:
                cap = cv2.VideoCapture(os.path.join(dataset_videos, video))
                to_capture = 50
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                toskip = math.floor(frame_count / to_capture)
                if toskip == 0:
                    toskip = 1

                frame_num = 0
                count = 1
                while (cap.isOpened()):
                    if len(os.listdir(os.path.join(images_dataset, name, video, folder))) == 50:
                        break
                    else:
                        ret, frame = cap.read()
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        frame_num = frame_num + toskip
                        if ret:
                            # Display the resulting frame
                            new_height = int(frame.shape[0] * size / 100)
                            new_width = int(frame.shape[1] * size / 100)
                            new_dimensions = (new_width, new_height)
                            output = cv2.resize(frame, new_dimensions)
                            # cv2.imshow('Frame', output)
                            os.chdir(os.path.join(images_dataset, name, video, folder))
                            if os.path.exists((str(count) + '.jpg')):
                                os.remove((str(count) + '.jpg'))
                            cv2.imwrite((str(count) + '.jpg'), output)

                            count = count + 1

                            if cv2.waitKey(25) & 0xFF == ord('q'):
                                break
                            # Break the loop
                        else:
                            cap.release()
                            break
                cap.release()
                cv2.destroyAllWindows()


# data augmentation function to rotate/skew passed frame
def SkewFramesCombo(angle, inputImg):
    try:
        img = inputImg
        (h, w) = img.shape[:2]
        rotpoint = (w // 2, h // 2)
        rotmat = cv2.getRotationMatrix2D(rotpoint, angle, 1.0)
        dim = (w, h)
        newFrame = cv2.warpAffine(img, rotmat, dim)
        return newFrame

        # cv2.imshow('Frame', newFrame)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
    except AttributeError:
        print('AttributeError in SkewFramesCombo func')
        return inputImg


# data augmentation function to rescale and rotate/skew frames
def ResizeSkewCombo(size, categories, images_dataset, dataset_videos):
    # all_videos = os.listdir(dataset_videos)  # list of actual videos

    folder = 0
    if size == 50:
        folder = 16
    elif size == 150:
        folder = 24
    elif size == 200:
        folder = 32
    elif size == 250:
        folder = 40
    if size == 300:
        folder = 48
    print('actions: ', categories)

    for name in categories:
        print("Thread ", threading.get_native_id(), " with size arg ", str(size),
              " executing ResizeSkewCombo function, now in class: ", name)

        selected_videos = os.listdir(os.path.join(images_dataset, name))
        for video in selected_videos:

            cap = cv2.VideoCapture(os.path.join(dataset_videos, video))
            to_capture = 50
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            toskip = math.floor(frame_count / to_capture)
            if toskip == 0:
                toskip = 1

            frame_num = 0
            count = 1
            while (cap.isOpened()):
                ret, frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                frame_num = frame_num + toskip
                if ret:
                    new_height = int(frame.shape[0] * size / 100)
                    new_width = int(frame.shape[1] * size / 100)
                    new_dimensions = (new_width, new_height)
                    output = cv2.resize(frame, new_dimensions)
                    # cv2.imshow('Frame', output)

                    angles = [-5, 5, -10, 10, -15, 15, -20, 20]
                    for i in range(len(angles)):
                        if len(os.listdir(os.path.join(images_dataset, name, video, str(folder + i)))) == 50:
                            continue
                        newOutput = SkewFramesCombo(angles[i], output)
                        os.chdir(os.path.join(images_dataset, name, video, str(folder + i)))
                        if os.path.exists((str(count) + '.jpg')):
                            os.remove((str(count) + '.jpg'))
                            # continue
                        cv2.imwrite((str(count) + '.jpg'), newOutput)
                        # cv2.imshow('Frame', newOutput)

                    count = count + 1
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    cap.release()
                    break
            cap.release()
            cv2.destroyAllWindows()



def Augmentation1(actions, images_dataset, dataset_videos):
    t1 = threading.Thread(target=getOriginalFrames, args=(actions, images_dataset, dataset_videos,))
    t1.start()


def Augmentation2(images_dataset):
    t2 = threading.Thread(target=Crop_Videos, args=(images_dataset,))
    t2.start()


def Augmentation3(angle, images_dataset):
    t3 = threading.Thread(target=SkewFrames, args=(angle, images_dataset,))
    t3.start()


def Augmentation4(size, images_dataset, dataset_videos):
    t4 = threading.Thread(target=ResizeFrames, args=(size, images_dataset, dataset_videos,))
    t4.start()


def Augmentation5(size, actions, images_dataset, dataset_videos):
    t5 = threading.Thread(target=ResizeSkewCombo, args=(size, actions, images_dataset, dataset_videos,))
    t5.start()



dataset_videos = r'D:\WLASL2000'  # orginal WLASL dataset videos
images_dataset = r'D:\ASL\NewDataSet'  # eligible words/classes

# original
# actions = os.listdir(images_dataset)
# length = math.floor(len(actions) / 8)
#
# first = actions[:length]
# second = actions[length:length * 2]
# third = actions[length * 2:length * 3]
# fourth = actions[length * 3:length * 4]
# fifth = actions[length * 4:length * 5]
# sixth = actions[length * 5:length * 6]
# seventh = actions[length * 6:length * 7]
# eighth = actions[length * 7:]




if __name__ == '__main__':
    pass
    # p1 = multiprocessing.Process(target=Augmentation1, args=(first, images_dataset, dataset_videos,))
    # p1.start()
    # p2 = multiprocessing.Process(target=Augmentation1, args=(second, images_dataset, dataset_videos,))
    # p2.start()
    # p3 = multiprocessing.Process(target=Augmentation1, args=(third, images_dataset, dataset_videos,))
    # p3.start()
    # p4 = multiprocessing.Process(target=Augmentation1, args=(fourth, images_dataset, dataset_videos,))
    # p4.start()
    # p5 = multiprocessing.Process(target=Augmentation1, args=(fifth, images_dataset, dataset_videos,))
    # p5.start()
    # p6 = multiprocessing.Process(target=Augmentation1, args=(sixth, images_dataset, dataset_videos,))
    # p6.start()
    # p7 = multiprocessing.Process(target=Augmentation1, args=(seventh, images_dataset, dataset_videos,))
    # p7.start()
    # p8 = multiprocessing.Process(target=Augmentation1, args=(eighth, images_dataset, dataset_videos,))
    # p8.start()

    # crop
    #     p2 = multiprocessing.Process(target=Augmentation2, args=(images_dataset, ))
    #     p2.start()

    # skew
    #     p3 = multiprocessing.Process(target=Augmentation3, args=(-5, images_dataset, ))
    #     p4 = multiprocessing.Process(target=Augmentation3, args=(5, images_dataset, ))
    #     p5 = multiprocessing.Process(target=Augmentation3, args=(-10, images_dataset, ))
    #     p6 = multiprocessing.Process(target=Augmentation3, args=(10, images_dataset, ))
    #     p7 = multiprocessing.Process(target=Augmentation3, args=(-15, images_dataset, ))
    #     p8 = multiprocessing.Process(target=Augmentation3, args=(15, images_dataset, ))
    #     p9 = multiprocessing.Process(target=Augmentation3, args=(-20, images_dataset, ))
    #     p10 = multiprocessing.Process(target=Augmentation3, args=(20, images_dataset, ))
    #
    #     p3.start()
    #     p4.start()
    #     p5.start()
    #     p6.start()
    #     p7.start()
    #     p8.start()
    #     p9.start()
    #     p10.start()

    # resize
    #     p11 = multiprocessing.Process(target=Augmentation4, args=(50, images_dataset, dataset_videos, ))
    #     p12 = multiprocessing.Process(target=Augmentation4, args=(150, images_dataset, dataset_videos, ))
    #     p13 = multiprocessing.Process(target=Augmentation4, args=(200, images_dataset, dataset_videos, ))
    #     p14 = multiprocessing.Process(target=Augmentation4, args=(250, images_dataset, dataset_videos, ))
    #     p15 = multiprocessing.Process(target=Augmentation4, args=(300, images_dataset, dataset_videos, ))
    #
    #     p11.start()
    #     p12.start()
    #     p13.start()
    #     p14.start()
    #     p15.start()

    # resize-skew combo
    #     p16 = multiprocessing.Process(target=Augmentation5, args=(50, first, images_dataset, dataset_videos))
    #     p17 = multiprocessing.Process(target=Augmentation5, args=(50, second, images_dataset, dataset_videos))
    #     p18 = multiprocessing.Process(target=Augmentation5, args=(50, third, images_dataset, dataset_videos))
    #     p19 = multiprocessing.Process(target=Augmentation5, args=(50, fourth, images_dataset, dataset_videos))
    #     p20 = multiprocessing.Process(target=Augmentation5, args=(50, fifth, images_dataset, dataset_videos))
    #     p21 = multiprocessing.Process(target=Augmentation5, args=(50, sixth, images_dataset, dataset_videos))
    #     p22 = multiprocessing.Process(target=Augmentation5, args=(50, seventh, images_dataset, dataset_videos))
    #     p23 = multiprocessing.Process(target=Augmentation5, args=(50, eighth, images_dataset, dataset_videos))
    #
    #     p16.start()
    #     p17.start()
    #     p18.start()
    #     p19.start()
    #     p20.start()
    #     p21.start()
    #     p22.start()
    #     p23.start()

    # p24 = multiprocessing.Process(target=Augmentation5, args=(150, first, images_dataset, dataset_videos))
    # p25 = multiprocessing.Process(target=Augmentation5, args=(150, second, images_dataset, dataset_videos))
    # p26 = multiprocessing.Process(target=Augmentation5, args=(150, third, images_dataset, dataset_videos))
    # p27 = multiprocessing.Process(target=Augmentation5, args=(150, fourth, images_dataset, dataset_videos))
    # p28 = multiprocessing.Process(target=Augmentation5, args=(150, fifth, images_dataset, dataset_videos))
    # p29 = multiprocessing.Process(target=Augmentation5, args=(150, sixth, images_dataset, dataset_videos))
    # p30 = multiprocessing.Process(target=Augmentation5, args=(150, seventh, images_dataset, dataset_videos))
    # p31 = multiprocessing.Process(target=Augmentation5, args=(150, eighth, images_dataset, dataset_videos))
    #
    # p24.start()
    # p25.start()
    # p26.start()
    # p27.start()
    # p28.start()
    # p29.start()
    # p30.start()
    # p31.start()

    # p32 = multiprocessing.Process(target=Augmentation5, args=(200, first, images_dataset, dataset_videos))
    # p33 = multiprocessing.Process(target=Augmentation5, args=(200, second, images_dataset, dataset_videos))
    # p34 = multiprocessing.Process(target=Augmentation5, args=(200, third, images_dataset, dataset_videos))
    # p35 = multiprocessing.Process(target=Augmentation5, args=(200, fourth, images_dataset, dataset_videos))
    # p36 = multiprocessing.Process(target=Augmentation5, args=(200, fifth, images_dataset, dataset_videos))
    # p37 = multiprocessing.Process(target=Augmentation5, args=(200, sixth, images_dataset, dataset_videos))
    # p38 = multiprocessing.Process(target=Augmentation5, args=(200, seventh, images_dataset, dataset_videos))
    # p39 = multiprocessing.Process(target=Augmentation5, args=(200, eighth, images_dataset, dataset_videos))
    #
    # p32.start()
    # p33.start()
    # p34.start()
    # p35.start()
    # p36.start()
    # p37.start()
    # p38.start()
    # p39.start()

    # p40 = multiprocessing.Process(target=Augmentation5, args=(250, first, images_dataset, dataset_videos))
    # p41 = multiprocessing.Process(target=Augmentation5, args=(250, second, images_dataset, dataset_videos))
    # p42 = multiprocessing.Process(target=Augmentation5, args=(250, third, images_dataset, dataset_videos))
    # p43 = multiprocessing.Process(target=Augmentation5, args=(250, fourth, images_dataset, dataset_videos))
    # p44 = multiprocessing.Process(target=Augmentation5, args=(250, fifth, images_dataset, dataset_videos))
    # p45 = multiprocessing.Process(target=Augmentation5, args=(250, sixth, images_dataset, dataset_videos))
    # p46 = multiprocessing.Process(target=Augmentation5, args=(250, seventh, images_dataset, dataset_videos))
    # p47 = multiprocessing.Process(target=Augmentation5, args=(250, eighth, images_dataset, dataset_videos))
    #
    # p40.start()
    # p41.start()
    # p42.start()
    # p43.start()
    # p44.start()
    # p45.start()
    # p46.start()
    # p47.start()

    # p48 = multiprocessing.Process(target=Augmentation5, args=(300, first, images_dataset, dataset_videos))
    # p49 = multiprocessing.Process(target=Augmentation5, args=(300, second, images_dataset, dataset_videos))
    # p50 = multiprocessing.Process(target=Augmentation5, args=(300, third, images_dataset, dataset_videos))
    # p51 = multiprocessing.Process(target=Augmentation5, args=(300, fourth, images_dataset, dataset_videos))
    # p52 = multiprocessing.Process(target=Augmentation5, args=(300, fifth, images_dataset, dataset_videos))
    # p53 = multiprocessing.Process(target=Augmentation5, args=(300, sixth, images_dataset, dataset_videos))
    # p54 = multiprocessing.Process(target=Augmentation5, args=(300, seventh, images_dataset, dataset_videos))
    # p55 = multiprocessing.Process(target=Augmentation5, args=(300, eighth, images_dataset, dataset_videos))
    #
    # p48.start()
    # p49.start()
    # p50.start()
    # p51.start()
    # p52.start()
    # p53.start()
    # p54.start()
    # p55.start()
