import glob
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import visualize
import torch
import albumentations as A


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


class ApplyTransform(Dataset):

    def __init__(self, dataset,
                 augmentation=None,
                 preprocessing=None,):
        self.dataset = dataset
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.dataset)


class BalloonDatasetSegmentation(Dataset):
    CLASSES = ['balloon']

    @staticmethod
    def minimal_transformations(input_shape):
        transform = [
            A.Resize(height=input_shape[0], width=input_shape[1], p=1)
        ]
        return A.Compose(transform)

    def __init__(self,
                 folder_path,
                 input_size=(512, 512),
                 classes=None,):
        super(BalloonDatasetSegmentation, self).__init__()

        self.img_files = glob.glob(os.path.join(folder_path, '*.jpg'))
        self.mask_files = glob.glob(
            os.path.join(folder_path, 'masks', '*.png'))
        self.class_values = [self.CLASSES.index(
            cls.lower()) for cls in classes]
        self.input_size = input_size

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=2)

        # apply minimal_transforms
        sample = self.minimal_transformations(self.input_size)(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.img_files)


class BalloonLoaders:

    @staticmethod
    def train_test_split(dataset, train_size_perc=0.8, ):

        train_size = int(train_size_perc * len(dataset))
        test_size = len(dataset) - train_size

        return torch.utils.data.random_split(dataset, [train_size, test_size])

    def __init__(self, preprocessing_fn, augmentation=None,
                 train_size_perc=0.8):

        self.train_dataset = BalloonDatasetSegmentation('balloon\\train',
                                                        classes=['balloon'],
                                                        )

        self.valid_dataset = BalloonDatasetSegmentation('balloon\\val',
                                                        classes=['balloon'],
                                                        )

<<<<<<< Updated upstream
        # Сначала сплитим данные, потом аугментации
        self.train_dataset, self.test_dataset = BalloonLoaders.train_test_split(
            self.train_dataset)

        # Аугментируем только тренировочные данные
        self.train_dataset = ApplyTransform(self.train_dataset,
                                            augmentation=augmentation,
                                            preprocessing=get_preprocessing(
                                                preprocessing_fn),
                                            )
        self.valid_dataset = ApplyTransform(self.valid_dataset,
                                            augmentation=None,
                                            preprocessing=get_preprocessing(
                                                preprocessing_fn),
                                            )
        self.test_dataset = ApplyTransform(self.test_dataset,
                                           augmentation=None,
                                           preprocessing=get_preprocessing(
                                               preprocessing_fn),
                                           )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=8, shuffle=True)
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=8, shuffle=False)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=8, shuffle=False)
=======
        self.train_dataset, self.test_dataset = BalloonLoaders.train_test_split(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=8, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=8, shuffle=False)
>>>>>>> Stashed changes

    def show_example(self):
        n = np.random.choice(len(self.train_dataset))
        image, mask = self.train_dataset[n]
        visualize(image=image.transpose((1, 2, 0)),
                  mask=mask.transpose((1, 2, 0)) * 255)
