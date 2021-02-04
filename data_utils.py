import torch
from torch.utils.data import Dataset
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import cv2

import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from skimage import color

#CIFAR10 = datasets.CIFAR10(root='.',train=True, download=True)


class mod_cifar(datasets.cifar.CIFAR10):
    def __init__(self, transform, train = True):
        super(mod_cifar, self).__init__(root='.', train=train)
        self.transform = transform

    @staticmethod
    def get_rbg_from_lab(gray_imgs: np.ndarray = None,
                         ab_imgs: np.ndarray = None, n=10):
        images_list = []
        for i in range(0, n):
            image_gray = np.expand_dims((gray_imgs[0:n:][i].squeeze().detach()
                                         .cpu().numpy())[:, :],axis=2)
            image_ab = (np.transpose(ab_imgs[0:n:][i].squeeze().detach()
                                     .cpu().numpy(), (1, 2, 0)))
            LAB = np.concatenate([image_gray, image_ab], axis=2)
            RGB = color.lab2rgb(LAB)
            images_list.append(RGB)
        images_list = np.array(images_list)
        return images_list

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgs_for_input = cv2.resize(self.data[idx], (64,64))
        lab = color.rgb2lab(imgs_for_input)

        sample = {'image_grey': lab[:,:,0],
                  'image_color': lab[:,:,0],
                  'img_ab': lab[:,:,1:]}

        if self.transform:
            sample = self.transform(sample)
        return sample


class GrayColorImg(Dataset):
    """gery color dataset"""
    @staticmethod
    def pipe_line_img(gray_scale_imgs: np.ndarray = None, batch_size=1,
                      preprocess_f=preprocess_input):
        imgs = np.zeros((batch_size, 224, 224, 3))
        for i in range(0, 3):
            imgs[:batch_size, :, :, i] = gray_scale_imgs
        return preprocess_f(imgs)

    @staticmethod
    def get_rbg_from_lab_one(gray_imgs: np.ndarray = None,
                             ab_imgs: np.ndarray = None):
        # create an empty array to store images
        imgs = np.zeros((1, 224, 224, 3))

        imgs[:, :, :, 0] = gray_imgs
        imgs[:, :, :, 1:] = ab_imgs

        # convert all the images to type unit8
        imgs = imgs.astype("uint8")

        # create a new empty array
        imgs_ = []

        for i in range(0, 1):
            imgs_.append(cv2.cvtColor(imgs[i], cv2.COLOR_LAB2RGB))

        # convert the image matrix into a numpy array
        imgs_ = np.array(imgs_)

        # print(imgs_.shape)

        return imgs_

    @staticmethod
    def get_rbg_from_lab(gray_imgs: np.ndarray = None,
                         ab_imgs: np.ndarray = None, n=10):
        # create an empty array to store images
        imgs = np.zeros((n, 224, 224, 3))

        imgs[:, :, :, 0] = gray_imgs[0:n:]
        imgs[:, :, :, 1:] = ab_imgs[0:n:]

        # convert all the images to type unit8
        imgs = imgs.astype("uint8")

        # create a new empty array
        imgs_ = []

        for i in range(0, n):
            imgs_.append(cv2.cvtColor(imgs[i], cv2.COLOR_LAB2RGB))

        # convert the image matrix into a numpy array
        imgs_ = np.array(imgs_)

        # print(imgs_.shape)

        return imgs_

    def __init__(self, path_file_input: str = '', path_file_out: str = '',
                 transform: transforms.Compose = None):
        self.data_input = np.load(path_file_input)
        self.data_output = np.load(path_file_out)
        self.transform = transform

    def __len__(self):
        return len(self.data_output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgs_for_input = self.data_input[idx]
        output_image = self.get_rbg_from_lab_one(self.data_input[idx],
                                            self.data_output[idx])
        sample = {'image_grey': imgs_for_input,
                  'image_color': output_image,
                  'img_ab': self.data_output[idx]}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_grey, image_color, image_ab = sample['image_grey'], sample[
            'image_color'], sample['img_ab']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image_grey = np.expand_dims(np.repeat(np.expand_dims(image_grey, axis=0)
                                              , repeats=3, axis=0), axis=0)
        #image_color = image_color.transpose((0, 3, 1, 2))
        image_ab = np.expand_dims(image_ab, axis=0).transpose((0, 3, 1, 2))

        return {'image_grey': torch.Tensor(image_grey),
                'image_color': torch.Tensor(image_color),
                'img_ab': torch.Tensor(image_ab)}


if __name__ == '__main__':
    composed = transforms.Compose([ToTensor()])
    aa = mod_cifar(transform=composed)
    out = aa.__getitem__(1)
    aa.get_rbg_from_lab_one
    image_grey = np.expand_dims( np.transpose(out['image_grey'].squeeze(),(1,2,0))[:,:,0],axis=2)
    image_ab = (np.transpose(out['img_ab'].squeeze(), (1, 2, 0)) )
    LAB= np.concatenate([image_grey, image_ab],axis=2)
    RGB = color.lab2rgb(LAB)
    plt.imshow(RGB )
    plt.figure(1)
    plt.imshow()
    plt.show()
    plt.figure(1)
    plt.imshow()
    plt.show()

