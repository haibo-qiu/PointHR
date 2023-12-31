import os
import json
import torch
import pickle
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset

from pcr.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose

@DATASETS.register_module()
class ShapeNetPartDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 num_points=2048,
                 presample=False,
                 loop=1):
        super(ShapeNetPartDataset, self).__init__()
        self.data_root = data_root
        self.npoints = num_points
        self.presample = presample
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.cache = {}

        # load categories file
        self.categories = []
        self.category2part = {'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11],
                              'Chair': [12, 13, 14, 15], 'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21],
                              'Knife': [22, 23], 'Lamp': [24, 25, 26, 27], 'Laptop': [28, 29],
                              'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37], 'Pistol': [38, 39, 40],
                              'Rocket': [41, 42, 43], 'Skateboard': [44, 45, 46], 'Table': [47, 48, 49]}
        self.token2category = {}
        with open(os.path.join(self.data_root, 'synsetoffset2category.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.token2category[ls[1]] = len(self.categories)
                self.categories.append(ls[0])

        self.cls2parts = []
        for cls in self.categories:
            idx = self.category2part[cls]
            self.cls2parts.append(idx)

        if test_mode:
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        # load data list
        if isinstance(self.split, str):
            self.data_list = self.load_data_list(self.split)
        elif isinstance(self.split, list):
            self.data_list = []
            for s in self.split:
                self.data_list += self.load_data_list(s)
        else:
            raise NotImplementedError

        logger = get_root_logger()
        if presample:
            filename = os.path.join(data_root, 'processed', f'{split}_{num_points}_fps.pkl')
            assert os.path.exists(filename)
            with open(filename, 'rb') as f:
                self.data, self.cls = pickle.load(f)
                logger.info("{} load successfully".format(filename))

        logger.info("Totally {} x {} samples in {} set.".format(len(self.data_list), self.loop, split))

    def load_data_list(self, split):
        split_file = os.path.join(self.data_root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        if not os.path.isfile(split_file):
            raise (RuntimeError("Split file do not exist: " + split_file + "\n"))
        with open(split_file, 'r') as f:
            # drop "shape_data/" and append ".txt"
            data_list = [os.path.join(self.data_root, data[11:] + '.txt') for data in json.load(f)]
        return data_list

    def prepare_train_data(self, idx):
        # load data
        data_idx = idx % len(self.data_list)
        if data_idx in self.cache:
            coord, norm, label, cls_token = self.cache[data_idx]
        else:
            if not self.presample:
                data = np.loadtxt(self.data_list[data_idx]).astype(np.float32)
                choice = np.random.choice(len(data), self.npoints, replace=True)
                data = data[choice]
                cls_token = self.token2category[os.path.basename(os.path.dirname(self.data_list[data_idx]))]
            else:
                data, cls_token = self.data[data_idx], self.cls[data_idx]

            coord, norm, label = data[:, :3], data[:, 3:6], data[:, 6].astype(np.int32)
            self.cache[data_idx] = (coord, norm, label, cls_token)

        data_dict = dict(coord=coord, norm=norm, label=label, cls_token=cls_token)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_idx = idx % len(self.data_list)
        if not self.presample:
            data = np.loadtxt(self.data_list[data_idx]).astype(np.float32)[:self.npoints]
            cls_token = self.token2category[os.path.basename(os.path.dirname(self.data_list[data_idx]))]
        else:
            data, cls_token = self.data[data_idx], self.cls[data_idx]

        coord, norm, label = data[:, :3], data[:, 3:6], data[:, 6].astype(np.int32)

        data_dict = dict(coord=coord, norm=norm, cls_token=cls_token)
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(
                self.post_transform(aug(deepcopy(data_dict)))
            )
        return data_dict_list, label

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        return os.path.basename(self.data_list[data_idx]).split(".")[0]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
