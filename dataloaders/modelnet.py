"""
Dataset classes for ModelNet40 dataset.

There are so many versions of ModelNet40.
"""
import os
import random

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

        """
        I'm not sure whether normalize the point cloud or sample randomly first.
        """

        # normalize into a sphere whose radius is 1
        if self.normalize:
            point_cloud = ModelNet40v1.pc_normalize(point_cloud)

        # select self.npoints from the original point cloud randomly
        choice = np.random.choice(len(point_cloud), self.npoints, replace=True)
        point_cloud = point_cloud[choice, :]
        normal = normal[choice, :]

        # data augmentation - random rotation and random jitter
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_cloud[:, [0, 2]] = point_cloud[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_cloud += np.random.normal(0, 0.02, size=point_cloud.shape)  # random jitter
        
        # whether return the normal
        if self.normal:
            return point_cloud, label, normal
        else:
            return point_cloud, label
    
    @staticmethod
    def pc_normalize( pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __len__(self):
        return len(self.pcs)


if __name__ == '__main__':
    DATA_PATH = '/home/rico/Workspace/Dataset/modelnet/modelnet40_ply_hdf5_2048'

    train_dataset = ModelNet40v1(root=DATA_PATH, split='train', data_augmentation=False, npoints=2048)
    test_dataset = ModelNet40v1(root=DATA_PATH, split='test', data_augmentation=False, npoints=2048)
    print("{} training data and {} testing data".format(len(train_dataset), len(test_dataset)))
    
    point_cloud, label = train_dataset[random.randint(0, len(train_dataset))]
    print(point_cloud.shape, label.shape)
    print(label[0])

    import sys
    sys.path.append('..')
    from visualization.visualization import show_point_cloud
    show_point_cloud(point_cloud)
