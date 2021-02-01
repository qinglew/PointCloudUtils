"""
Dataset classes for ModelNet40 dataset.

There are so many versions of ModelNet40.
"""
import os
import random
import pprint

import h5py
import numpy as np
import torch.utils.data as data


class ModelNet40v1(data.Dataset):
    """
    ModelNet40 dataset class for the file "modelnet40_ply_hdf5_2048.zip" which has 435.2MB in Ubuntu.
    The files by unzip command is blew:
        - shape_names.txt
        - ply_data_train_2_id2file.json
        - train_files.txt
        - test_files.txt
        - ply_data_train_3_id2file.json
        - ply_data_test0.h5
        - ply_data_train3.h5
        - ply_data_test_1_id2file.json
        - ply_data_train0.h5
        - ply_data_train2.h5
        - ply_data_test_0_id2file.json
        - ply_data_train_0_id2file.json
        - ply_data_train4.h5
        - ply_data_test1.h5
        - ply_data_train_1_id2file.json
        - ply_data_train1.h5
        - ply_data_train_4_id2file.json
    
    The label for the point cloud is an integer. If you want to see the corresponding category name, please
    see the file 'modelnet40category2id.txt' in current directory.

    Maybe this implementation can be optimized into a faster way.

    Attributes:
        root [str]: the root directory of the modelnet40 dataset
        npoints [int]: the number of points sampled from the original point cloud randomly
        normalize [bool]: normalize the point cloud into a sphere whose radius is 1
        data_autmentation [bool]: random ratate and random jitter the point cloud
        normal [bool]: need to get the normals or not
    """
    def __init__(self, root, npoints=1024, split='train', normalize=True, data_augmentation=True, normal=False):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.normalize = normalize
        self.data_augmentation = data_augmentation
        self.normal = normal
        self.cat2id = {}

        # load classname and class id
        with open('datasets/modelnet40category2id.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat2id[ls[0]] = int(ls[1])
        self.id2cat = {v: k for k, v in self.cat2id.items()}

        # find all .h5 files
        with open(os.path.join(self.root, '{}_files.txt'.format(split)), 'r') as f:
            data_files = [os.path.join(self.root, line.strip().split('/')[-1]) for line in f]

        # load data from .h5 files
        point_clouds, labels, normals = [], [], []
        for filename in data_files:
            with h5py.File(filename, 'r') as data_file:
                point_clouds.append(np.array(data_file['data']))
                labels.append(np.array(data_file['label']))
                normals.append(np.array(data_file['normal']))
        self.pcs = np.concatenate(point_clouds, axis=0)
        self.lbs = np.concatenate(labels, axis=0)
        self.nms = np.concatenate(normals, axis=0)
    
    def __getitem__(self, index):
        point_cloud = self.pcs[index]
        label = self.lbs[index]
        normal = self.nms[index]
        classname = self.id2cat[label[0]]

        """
        I'm not sure whether normalize the point cloud or sample randomly first.
        """

        # select self.npoints from the original point cloud randomly
        choice = np.random.choice(len(point_cloud), self.npoints, replace=True)
        point_cloud = point_cloud[choice, :]
        normal = normal[choice, :]

        # normalize into a sphere whose radius is 1
        if self.normalize:
            point_cloud = point_cloud - np.mean(point_cloud, axis=0)
            dist = np.max(np.sqrt(np.sum(point_cloud  ** 2, axis=1)))
            point_cloud = point_cloud / dist

        # data augmentation - random rotation and random jitter
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_cloud[:, [0, 2]] = point_cloud[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_cloud += np.random.normal(0, 0.02, size=point_cloud.shape)  # random jitter
        
        # whether return the normal
        if self.normal:
            return point_cloud, label, normal, classname
        else:
            return point_cloud, label, classname

    def __len__(self):
        return len(self.pcs)


class ModelNet40v2(data.Dataset):
    """
    ModelNet40 dataset class for the file "modelnet40_hdf5_2048.zip" which has 204.0MB in Ubuntu.
    This file do not contains the normals.
    The files by unzip command is blew:
        - 'shape_names.txt',
        - 'test1.h5',
        - 'train2_id2file.json',
        - 'train_files.txt',
        - 'test_files.txt',
        - 'train1_id2name.json',
        - 'train0_id2file.json',
        - 'train3.h5',
        - 'test1_id2file.json',
        - 'train4_id2file.json',
        - 'test0.h5',
        - 'train1_id2file.json',
        - 'train1.h5',
        - 'train2_id2name.json',
        - 'test1_id2name.json',
        - 'test0_id2file.json',
        - 'train3_id2file.json',
        - 'train4_id2name.json',
        - 'train0_id2name.json',
        - 'train3_id2name.json',
        - 'train0.h5',
        - 'test0_id2name.json',
        - 'train4.h5',
        - 'train2.h5'
    
    The label for the point cloud is an integer. If you want to see the corresponding category name, please
    see the file 'modelnet40category2id.txt' in current directory.

    Maybe this implementation can be optimized into a faster way.

    Attributes:
        root [str]: the root directory of the modelnet40 dataset
        npoints [int]: the number of points sampled from the original point cloud randomly
        normalize [bool]: normalize the point cloud into a sphere whose radius is 1
        data_autmentation [bool]: random ratate and random jitter the point cloud
    """
    def __init__(self, root, npoints=1024, split='train', normalize=True, data_augmentation=True):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.normalize = normalize
        self.data_augmentation = data_augmentation
        self.cat2id = {}

        # load classname and class id
        with open('datasets/modelnet40category2id.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat2id[ls[0]] = int(ls[1])
        self.id2cat = {v: k for k, v in self.cat2id.items()}

        # find all .h5 files
        with open(os.path.join(self.root, '{}_files.txt'.format(split)), 'r') as f:
            data_files = [os.path.join(self.root, line.strip().split('/')[-1]) for line in f]

        # load data from .h5 files
        point_clouds, labels = [], []
        for filename in data_files:
            with h5py.File(filename, 'r') as data_file:
                point_clouds.append(np.array(data_file['data']))
                labels.append(np.array(data_file['label']))
        self.pcs = np.concatenate(point_clouds, axis=0)
        self.lbs = np.concatenate(labels, axis=0)
    
    def __getitem__(self, index):
        point_cloud = self.pcs[index]
        label = self.lbs[index]
        classname = self.id2cat[label[0]]

        # select self.npoints from the original point cloud randomly
        choice = np.random.choice(len(point_cloud), self.npoints, replace=True)
        point_cloud = point_cloud[choice, :]

        # normalize into a sphere whose radius is 1
        if self.normalize:
            point_cloud = point_cloud - np.mean(point_cloud, axis=0)
            dist = np.max(np.sqrt(np.sum(point_cloud  ** 2, axis=1)))
            point_cloud = point_cloud / dist

        # data augmentation - random rotation and random jitter
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_cloud[:, [0, 2]] = point_cloud[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_cloud += np.random.normal(0, 0.02, size=point_cloud.shape)  # random jitter

        return point_cloud, label, classname

    def __len__(self):
        return len(self.pcs)


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    sys.path.append('..')
    from visualization.visualization import show_point_cloud

    # Testing for ModelNet40v1 -----------------------------------------------------------------------
    print('\033[32mTesting for the dataset file "modelnet40_ply_hdf5_2048.zip"------------------------\033[0m')
    DATA_PATH1 = '/home/rico/Workspace/Dataset/modelnet/modelnet40_ply_hdf5_2048'

    train_dataset = ModelNet40v1(root=DATA_PATH1, split='train', data_augmentation=False, npoints=2048)
    test_dataset = ModelNet40v1(root=DATA_PATH1, split='test', data_augmentation=False, npoints=2048)
    
    print("{} training data and {} testing data".format(len(train_dataset), len(test_dataset)))
    
    point_cloud, label, classname = train_dataset[random.randint(0, len(train_dataset))]
    print(point_cloud.shape, label.shape)
    print(label[0], classname)
    print()

    show_point_cloud(point_cloud)
    # -------------------------------------------------------------------------------------------------

    # Testing for ModelNet40v2 ------------------------------------------------------------------------
    print('\033[32mTesting for the dataset file "modelnet40_hdf5_2048.zip"-----------------------------\033[0m')
    DATA_PATH2 = '/home/rico/Workspace/Dataset/modelnet/modelnet40_hdf5_2048'
    
    train_dataset = ModelNet40v2(root=DATA_PATH2, split='train', data_augmentation=False, npoints=2048)
    test_dataset = ModelNet40v2(root=DATA_PATH2, split='test', data_augmentation=False, npoints=2048)
    
    print("{} training data and {} testing data".format(len(train_dataset), len(test_dataset)))
    
    point_cloud, label, classname = train_dataset[random.randint(0, len(train_dataset))]
    print(point_cloud.shape, label.shape)
    print(label[0], classname)
    print()

    show_point_cloud(point_cloud)
    # -------------------------------------------------------------------------------------------------
