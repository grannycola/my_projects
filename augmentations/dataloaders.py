import glob
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import get_preprocessing, visualize
import torch
import albumentations as A


class BalloonDatasetSegmentation(Dataset):
    CLASSES = ['balloon']

    def __init__(self,
                 folder_path,
                 classes=None,
                 augmentation=None,
                 preprocessing=None):
        super(BalloonDatasetSegmentation, self).__init__()

        self.img_files = glob.glob(os.path.join(folder_path, '*.jpg'))
        self.mask_files = glob.glob(
            os.path.join(folder_path, 'masks', '*.png'))
        self.class_values = [self.CLASSES.index(
            cls.lower()) for cls in classes]
        self.preprocessing = preprocessing
        self.augmentation = augmentation

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=2)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.img_files)


class BalloonLoaders:
    @staticmethod
    def minimal_transformations():
        transform = [
            A.Resize(height=512, width=512, p=1)
        ]
        return A.Compose(transform)

    @staticmethod
    def train_test_split(dataset, train_size_perc=0.8, ):

        train_size = int(train_size_perc * len(dataset))
        test_size = len(dataset) - train_size

        return torch.utils.data.random_split(dataset, [train_size, test_size])

    def __init__(self, preprocessing_fn, augmentation=None,
                 train_size_perc=0.8):

        # Если аугментации не указаны, применить минимальные трансформации
        if not augmentation:
            augmentation = self.minimal_transformations()

        self.train_dataset = BalloonDatasetSegmentation('balloon\\train',
                                                        classes=['balloon'],
                                                        preprocessing=get_preprocessing(
                                                            preprocessing_fn),
                                                        augmentation=augmentation, )

        self.valid_dataset = BalloonDatasetSegmentation('balloon\\val',
                                                        classes=['balloon'],
                                                        preprocessing=get_preprocessing(
                                                            preprocessing_fn),
                                                        augmentation=self.minimal_transformations())

        self.train_dataset, self.test_dataset = BalloonLoaders.train_test_split(
            self.train_dataset)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=8, shuffle=True)
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=8, shuffle=False)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=8, shuffle=False)

    def show_example(self):
        n = np.random.choice(len(self.train_dataset))
        image, mask = self.train_dataset[n]
        visualize(image=image.transpose((1, 2, 0)),
                  mask=mask.transpose((1, 2, 0)) * 255)
