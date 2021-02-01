"""
Visualization for point cloud using matploblib.
"""
import random
import sys

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(".")
from dataset.shapenet import ShapeNetPartDataset


def show_point_cloud_part(point_cloud, seg, axis=False):
    """show different parts of point cloud in different color

    Args:
        point_cloud (np.ndarray): the coordinates of point cloud
        seg (np.ndarray): the label of every point corresponding to the point cloud
        axis (bool, optional): Hid the coordinate of the matplotlib. Defaults to False
    """
    colors = ['red', 'green', 'blue', 'yellow', 'pink', 'cyan']
    parts = np.unique(seg)
    ax = plt.figure().add_subplot(projection='3d')
    ax._axis3don = axis
    for i in parts:
        index = (seg == i)
        ax.scatter(xs=point_cloud[index, 0], ys=point_cloud[index, 1], zs=point_cloud[index, 2], c=colors[i])
    plt.show()


def show_point_cloud(point_cloud, axis=False):
    """visual a point cloud

    Args:
        point_cloud (np.ndarray): the coordinates of point cloud
        axis (bool, optional): Hid the coordinate of the matplotlib. Defaults to False.
    """
    ax = plt.figure().add_subplot(projection='3d')
    ax._axis3don = axis
    ax.scatter(xs=point_cloud[:, 0], ys=point_cloud[:, 1], zs=point_cloud[:, 2])
    plt.show()


if __name__ == '__main__':
    dataset_path = "/home/rico/Workspace/Dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0"
    train_dataset = ShapeNetPartDataset(root=dataset_path, npoints=4096, classification=False)
    num = len(train_dataset)
    point_cloud, seg = train_dataset[random.randint(0, num)]
    print(seg.unique())

    show_point_cloud_part(point_cloud, seg)
    show_point_cloud(point_cloud)
