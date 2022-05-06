import cv2
import numpy as np
import os


def enumerate_files(dirs, path='All_gray_1_32_32/', n_poses=3, n_samples=20):
    filenames, targets = [], []
    for p in dirs:
        for n in range(n_poses):
            for j in range(3):
                dir_name = path+p+'/000'+str(n*3+j)+'/'
                for s in range(n_samples):
                    d = dir_name + '%04d/' % s
                    for f in os.listdir(d):
                        if f.endswith('jpg'):
                            filenames += [d + f]
                            targets.append(n)
    return filenames, targets


def read_images(files):
    imgs = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        imgs.append(img)
    return imgs


def read_datasets(datasets):
    files, labels = enumerate_files(datasets)
    list_of_arrays = read_images(files)
    return np.array(list_of_arrays), labels


if __name__ == "__main__":
    train_sets = ['Set1', 'Set2', 'Set3']
    test_sets = ['Set4', 'Set5']
    trn_array, trn_labels = read_datasets(train_sets)
    tst_array, tst_labels = read_datasets(test_sets)
    print(trn_array[0][0])
