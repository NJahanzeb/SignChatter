import math
import numpy as np
import os
import multiprocessing


# smaller function to sort files before concatenation
def sort_files(frames):
    finalFrames = []
    for i in range(len(frames)):
        finalFrames.append(int(frames[i].replace('.jpg.npy', '')))
    finalFrames.sort()

    for j in range(len(finalFrames)):
        finalFrames[j] = str(finalFrames[j]) + '.jpg.npy'
    return finalFrames


# function to read npy files of all frames and to concatenate and store them as a 3D array
def concat(actions, name):
    data_path = r"D:/ASL/NewKeyPoints"
    npy_path = f"E:/60_classes_sorted_concat/{name}"
    video_list = []
    i = 1
    for action in actions:
        videos = os.listdir(os.path.join(data_path, action))
        for video in videos:
            augs = os.listdir(os.path.join(data_path, action, video))
            # augs = augs[:16]
            # print("---------------------------------------------------------------------------------")
            # print('In the Video', video)
            # print("---------------------------------------------------------------------------------")
            for aug in augs:
                img = []
                frames = os.listdir(os.path.join(data_path, action, video, aug))                                #wheres the sort
                frames = sort_files(frames)

                # print("---------------------------------------------------------------------------------")
                # print('In the Augmentation ', aug)
                # print("---------------------------------------------------------------------------------")
                for frame in frames:
                    res = np.load(os.path.join(data_path, action, video, aug, frame))
                    img.append(res)
                video_list.append(img)
        print(os.getpid(), ' completed action: ', action)
        print('iteration: ', i)
        i += 1
    X = np.array(video_list)
    np.save(npy_path, X)


# function to generate and store labels for each video and all its augmentations
def labels():
    labels = []
    label_map = {label: num for num, label in enumerate(actions)}
    for action in actions:
        print(action)
        videos = os.listdir(os.path.join(os.getcwd(), action))
        for video in videos:
            augs = os.listdir(os.path.join(os.getcwd(), action, video))
            for aug in augs:
                labels.append(label_map[action])
    Y = np.array(labels)
    np.save(r"E:\25FramesArray\all_labels", Y)


# function to read and join smaller npy files into one big npy file
def merge(dir_path, save_name):
    # Initialize empty list to store arrays
    arrays = []
    print(os.listdir(dir_path))
    # Loop through each numpy file in directory
    for filename in os.listdir(dir_path):
        if filename.endswith('.npy'):
            # Load numpy file into array and append to list
            print(filename)
            file_path = os.path.join(dir_path, filename)
            arr = np.load(file_path)
            arrays.append(arr)

    # Concatenate arrays along first axis
    concatenated_array = np.concatenate(arrays, axis=0)

    # Save concatenated array to new numpy file
    save_path = f'E:/ASL/{save_name}.npy'
    np.save(save_path, concatenated_array)


if __name__ == '__main__':
    DATA_PATH = r"D:\ASL\NewKeyPoints"
    actions = os.listdir(DATA_PATH)
    actions = actions[:60]
    length = math.floor(len(actions) / 6)
    first = actions[:length]
    second = actions[length:length * 2]
    third = actions[length * 2:length * 3]
    fourth = actions[length * 3:length * 4]
    fifth = actions[length * 4:length * 5]
    sixth = actions[length * 5:]



    p1 = multiprocessing.Process(target=concat, args=(first, '1',))

    p2 = multiprocessing.Process(target=concat, args=(second, '2',))

    p3 = multiprocessing.Process(target=concat, args=(third, '3',))

    p4 = multiprocessing.Process(target=concat, args=(fourth, '4',))

    p5 = multiprocessing.Process(target=concat, args=(fifth, '5',))

    p6 = multiprocessing.Process(target=concat, args=(sixth, '6',))

    # p7 = multiprocessing.Process(target=concat, args=(seventh, '7',))


    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    # p7.start()



    # code to run after all the arrays have been concatinated

    # merge(r'D:\ASL\Final_Concatination', 'allarrays')
    # labels()
    # f=np.load(r'E:\ASL\allarrays.npy')
    # f1=np.load(r"E:\ASL\all_labels.npy")
    # print(f1)
    # print(f.shape)
    # print(f1.shape)
