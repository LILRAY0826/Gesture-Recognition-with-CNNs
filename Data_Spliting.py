import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from skimage import io


class CustomImageDataset(Dataset):
    def __init__(self,  csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = io.imread(img_path)
        label = torch.tensor(int(self.annotations.iloc[index, 2]))

        if self.transform:
            image = self.transform(image)

        return image, label


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
                            filename = d + f
                            filename = filename.replace("All_gray_1_32_32/", "")
                            filenames += [filename]
                            targets.append(n)

    return filenames, targets


def read_images(files):
    imgs = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        imgs.append(img)
    return imgs


def read_datasets(datasets, csv_name):
    files, labels = enumerate_files(datasets)
    dataframe = {"filename": files,
                 "label": labels}
    dataframe = pd.DataFrame(dataframe)
    dataframe.to_csv(csv_name)
    list_of_arrays = read_images(files)
    return np.array(list_of_arrays), labels


if __name__ == "__main__":
    train_sets = ['Set1', 'Set2', 'Set3']
    test_sets = ['Set4', 'Set5']
    trn_array, trn_labels = read_datasets(train_sets, csv_name="train_data.csv")
    tst_array, tst_labels = read_datasets(test_sets, csv_name="test_data.csv")
