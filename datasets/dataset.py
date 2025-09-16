from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
# from scipy.ndimage.interpolation import zoom
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
import cv2

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]

        # img = np.array(Image.open(img_path).convert('RGB'))

        img_BGR = cv2.imread(img_path, 1)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        img = np.array(img_RGB)

        
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        # kernel = np.ones((4, 4), np.uint8)
        img, msk = self.transformer((img, msk))
        # print(np.std(img,ddof=0))
        return img, msk

    def __len__(self):
        return len(self.data)

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class NPY_datasets_test(Dataset):
    # def __init__(self, path_Data, config, test=True):
    #     # super(NPY_datasets, self)
    #     super().__init__()
    #     if test:
    #         images_list = sorted(os.listdir(path_Data+'test/images/'))
    #         masks_list = sorted(os.listdir(path_Data+'test/masks/'))
    #         self.data = []
    #         for i in range(len(images_list)):
    #             img_path = path_Data+'test/images/' + images_list[i]
    #             mask_path = path_Data+'test/masks/' + masks_list[i]
    #             self.data.append([img_path, mask_path])
    #         self.transformer = config.train_transformer
    #     else:
    #         images_list = sorted(os.listdir(path_Data+'val/images/'))
    #         masks_list = sorted(os.listdir(path_Data+'val/masks/'))
    #         self.data = []
    #         for i in range(len(images_list)):
    #             img_path = path_Data+'val/images/' + images_list[i]
    #             mask_path = path_Data+'val/masks/' + masks_list[i]
    #             self.data.append([img_path, mask_path])
    #         self.transformer = config.test_transformer

    def __init__(self, path_Data, config, test=True):
        super().__init__()
        split = 'test' if test else 'val'
        # img_dir = os.path.join(path_Data, split, 'images')
        # msk_dir = os.path.join(path_Data, split, 'masks')
        img_dir = os.path.join(path_Data,  'images')
        msk_dir = os.path.join(path_Data,  'masks')
        img_map = _files_by_stem(img_dir, IMG_EXTS)
        msk_map = _files_by_stem(msk_dir, MSK_EXTS)
        common = sorted(set(img_map) & set(msk_map))
        if not common:
            raise RuntimeError(f"No matched image–mask pairs in {img_dir} and {msk_dir}.")
        self.data = [(img_map[s], msk_map[s]) for s in common]
        self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)


##################################

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
MSK_EXTS = IMG_EXTS

def _files_by_stem(dirpath, exts):
    files = []
    for f in os.listdir(dirpath):
        if f.startswith('.'):
            continue
        stem, ext = os.path.splitext(f)
        if ext.lower() in exts:
            files.append((stem, os.path.join(dirpath, f)))
    return {stem: path for stem, path in files}

####################################


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
        
    