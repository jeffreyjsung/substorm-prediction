import os
from torchvision import transforms
from PIL import Image, ImageOps
from PIL import UnidentifiedImageError
import torch
import torchvision
from dataHandler.logger import logger
import numpy as np


class BasicAsimDataSet(object):
    def __init__(self, path_list: [], index_list, size, mean, std):
        not_available = []
        for i, file in enumerate(path_list):
            if not os.path.exists(file):
                not_available.append(i)
        for i in sorted(not_available, reverse=True):
            del path_list[i]
            del index_list[i]
        self.dataset = path_list
        self.indices = index_list
        self.transform_nn = self.build_transform_nn(size, mean, std)

    def load_image(self, path):
        pass

    def __getitem__(self, idx):
        item = self.dataset[idx]
        index = self.indices[idx]
        return self.load_image(item), index

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def build_transform_nn(size, mean, std):
        transf = []
        transf.append(transforms.ToTensor())
        transf.append(transforms.Lambda(lambda x: x.squeeze()))
        transf.append(transforms.Lambda(lambda x: torch.stack([torch.zeros(x.shape),
                                                               x, torch.zeros(x.shape)])))
        if size <= 256:
            transf.append(transforms.Resize((size, size)))
        else:
            transf.append(torchvision.transforms.Pad(int((size - 256) / 2 + 1), fill=0,
                                                     padding_mode='constant'))
            transf.append(transforms.Resize((size, size)))
        transf.append(transforms.Lambda(lambda x: x / torch.max(x)))
        transf.append(transforms.Normalize(mean=mean, std=std))
        trf = transforms.Compose(transf)
        return trf


class OathDataSet(BasicAsimDataSet):
    def __init__(self, path_list: [], index_list: [], size, mean, std):
        super(OathDataSet, self).__init__(path_list, index_list, size, mean, std)

    def load_image(self, path):
        if not os.path.exists(path):
            return
        try:
            Image.open(path).verify()
        except (IOError, SyntaxError, UnidentifiedImageError):
            logger.error("Corrupted File")
            return None
        try:
            im = Image.open(path)
            im = np.array(im)
            im = im[:, :, 1]
            trim = self.transform_nn(im)
        except Exception as e:
            logger.error(e)
            logger.error("Can't transform image")
        return trim, path


class AsimDataSet(BasicAsimDataSet):
    def __init__(self, path_list: [], index_list):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.size = 224
        super(AsimDataSet, self).__init__(path_list, index_list, self.size, self.mean, self.std)
        not_available = []
        for i, file in enumerate(path_list):
            if not os.path.exists(file):
                not_available.append(i)
        for i in sorted(not_available, reverse=True):
            del path_list[i]
            del index_list[i]
        self.dataset = path_list
        self.indices = index_list
        self.im_max = 65535
        self.n = 128
        self.weights = self.build_weights()
        self.transform_im = self.build_transform_im()
        self.transform_hist = self.build_transform_hist()
        self.transform_nn = self.scaled_transform(window=0)

    def scaled_transform(self, window: int = 20):
        def image_weights():
            weights = {}
            for h in range(0, 750, 1):
                r = (h - 1) / 2
                x, y = (np.mgrid[:h, :h] - r)
                weight = 1 * ((x ** 2 + y ** 2) <= (r - window) ** 2)
                weights.update({h: weight})
            return weights

        def remove_upper_percentile(im, q, weight):
            boundary = np.percentile(im[np.where(weight)], q)
            im[im >= boundary] = boundary
            return im

        def remove_lower_percentile(im, q, weight):
            boundary = np.percentile(im[np.where(weight)], q)
            im[im <= boundary] = boundary
            return im

        def scale(im, weight):
            low = torch.min(im[np.where(weight)])
            high = torch.max(im[np.where(weight)])
            im = (im - low) / (high - low)
            return im

        def crop(im):
            return im[window:-window, window:-window]

        weights = image_weights()
        transf = []
        # Transform to Tensor, remove additional dimensions
        transf.append(transforms.ToTensor())
        transf.append(transforms.Lambda(lambda x: x.squeeze()))
        # Crop by setting everything outside a disk to zero
        transf.append(transforms.Lambda(
            lambda x: 1. * x * torch.tensor(weights.get(x.shape[0]))
        ))
        # Remove upper and lower percentile to remove background and oversaturation
        transf.append(transforms.Lambda(
            lambda x: remove_lower_percentile(x, .5, weights.get(x.shape[0]))
        ))
        transf.append(transforms.Lambda(
            lambda x: remove_upper_percentile(x, 99.5, weights.get(x.shape[0]))
        ))
        # Scale min and max between 0 and 1
        transf.append(transforms.Lambda(
            lambda x: scale(x, weights.get(x.shape[0]))
        ))
        # Crop zeroed-elements out of image
        if window > 0:
            transf.append(transforms.Lambda(
                lambda x: crop(x)
            ))
        # GrayLevel -> RGB
        transf.append(transforms.Lambda(
            lambda x: torch.stack([torch.zeros(x.shape), x, torch.zeros(x.shape)])
        ))
        # Resize and Normalize to Specs of Neural network
        transf.append(transforms.Resize((self.size, self.size)))
        transf.append(transforms.Normalize(
            mean=self.mean, std=self.std
        ))
        trf = transforms.Compose(transf)
        return trf

    @staticmethod
    def build_weights():
        weights = {}
        for h in range(251, 551, 1):
            r = (h - 1) / 2
            x, y = (np.mgrid[:h, :h] - r)
            weight = 1 * ((x ** 2 + y ** 2) <= (r + 0.5) ** 2)
            weight_here = {h**2: weight.ravel()}
            weights.update(weight_here)
        return weights

    def load_image(self, path):
        if not os.path.exists(path):
            return
        try:
            Image.open(path).verify()
        except (IOError, SyntaxError, UnidentifiedImageError):
            logger.error("Corrupted File")
            return None
        try:
            im = Image.open(path)
            im = ImageOps.grayscale(im)
            trim = self.transform_nn(im)
            im = self.transform_im(im)
            hist = self.transform_hist(im)
        except Exception as e:
            logger.error(e)
            logger.error("Can't transform image")
        return (im, trim, hist, path)

    def build_transform_im(self):
        transf = []
        transf.append(transforms.Lambda(lambda x: np.array(x)))
        transf.append(transforms.Lambda(lambda x: x.ravel()))
        transf.append(transforms.Lambda(lambda x: np.delete(x, self.weights[len(x)] == 0)))
        transf.append(transforms.Lambda(lambda x: np.expand_dims(x, 0)))
        transf.append(transforms.ToTensor())
        transf.append(transforms.Lambda(lambda x: (torch.ceil((x+1) / ((self.im_max+1)/self.n))-1)))
        trf = transforms.Compose(transf)
        return trf

    def build_transform_hist(self):
        transf = []
        transf.append(transforms.Lambda(lambda x: torch.histc(x, self.n, 0, self.n - 1)))
        trf = transforms.Compose(transf)
        return trf
