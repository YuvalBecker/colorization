import torch
import numpy as np
import cv2
from torchvision import transforms, datasets

from skimage import color

CIFAR10 = datasets.CIFAR10(root='.', train=True, download=True)


class mod_cifar(datasets.cifar.CIFAR10):
    def __init__(self, transform, train=True):
        super(mod_cifar, self).__init__(root='.', train=train)
        self.transform = transform

    @staticmethod
    def get_rbg_from_lab(gray_imgs: np.ndarray = None,
                         ab_imgs: np.ndarray = None, n=10):
        images_list = []
        for i in range(0, n):
            image_gray = np.expand_dims((gray_imgs[0:n:][i].squeeze().detach()
                                         .cpu().numpy())[:, :], axis=2)
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
        imgs_for_input = cv2.resize(self.data[idx], (64, 64))
        lab = color.rgb2lab(imgs_for_input)

        sample = {'image_grey': lab[:, :, 0],
                  'image_color': lab[:, :, 0],
                  'img_ab': lab[:, :, 1:]}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_grey, image_color, image_ab = sample['image_grey'], sample[
            'image_color'], sample['img_ab']

        image_grey = np.expand_dims(np.repeat(np.expand_dims(image_grey, axis=0)
                                              , repeats=3, axis=0), axis=0)
        image_ab = np.expand_dims(image_ab, axis=0).transpose((0, 3, 1, 2))

        return {'image_grey': torch.Tensor(image_grey),
                'image_color': torch.Tensor(image_color),
                'img_ab': torch.Tensor(image_ab)}


if __name__ == '__main__':
    composed = transforms.Compose([ToTensor()])
    aa = mod_cifar(transform=composed)
    out = aa.__getitem__(1)

