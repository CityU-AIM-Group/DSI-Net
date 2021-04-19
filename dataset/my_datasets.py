import numpy as np
import random
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import pickle


class CADCAPDataset(data.Dataset):
    def __init__(self, dataset_root, DATA_PKL, SIZE, data_type = 'train', mode = 'train'):
        self.dataset_root = dataset_root
        if os.path.exists(DATA_PKL):        
            with open(DATA_PKL, 'rb') as f:
                info = pickle.load(f)
        if data_type == 'train':
            self.data = info['train']
        else:
            self.data = info['test']
        assert (data_type == 'train' and mode == 'train') or (data_type == 'train' and mode == 'test') or (data_type == 'test' and mode == 'test'), print('mode setting error in dataset....')
        self.mode = mode
        self.train_augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=90, shear=5.729578),             
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(SIZE)
             ])
        self.train_gt_augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=90, shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(SIZE)
             ])
        
        self.test_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(SIZE)
             ])
        self.test_gt_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(SIZE)
             ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data[idx]
        image = Image.open(os.path.join(self.dataset_root, patient['image']))
        mask = Image.open(os.path.join(self.dataset_root, patient['mask'])).convert('1')
        label = patient['label']
        
        if self.mode == 'train':
            seed = np.random.randint(123456)
            random.seed(seed)
            image = self.train_augmentation(image)
            random.seed(seed)
            mask = self.train_gt_augmentation(mask)
        else:
            image = self.test_augmentation(image)
            mask = self.test_gt_augmentation(mask)             
                        
        image = np.array(image) / 255. 
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        mask = np.array(mask)
        mask = np.float32(mask > 0)
        name = patient['image'].split('.')[0].replace('/','_' ) 
        
        return image.copy(), mask.copy(), label, name






