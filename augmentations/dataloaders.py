import glob
import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class BalloonDatasetSegmentation(Dataset):
    CLASSES = ['balloon']

    def __init__(self,
                 folder_path,
                 classes=None,
                 augmentation=None,
                 preprocessing=None):
        super(BalloonDatasetSegmentation, self).__init__()

        self.img_files = glob.glob(os.path.join(folder_path, '*.jpg'))
        self.mask_files = glob.glob(os.path.join(folder_path, 'masks', '*.png'))

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
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
