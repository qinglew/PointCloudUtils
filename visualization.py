"""
Visualization for point cloud using matploblib.
"""
import random
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

from shapenet_part_loader import PartDataset


def show_point_cloud_part(point_cloud, seg, axis=False):
    """show different parts of point cloud in different color

    Args:
        point_cloud (np.ndarray): the coordinates of point cloud
        seg (np.ndarray): the label of every point corresponding to the point cloud
        axis (bool, optional): Hid the coordinate of the matplotlib. Defaults to False
    """
    colors = ['red', 'green', 'blue', 'yellow']
    parts = len(np.unique(seg))
    ax = plt.figure().add_subplot(projection='3d')
    ax._axis3don = axis
    for i in range(parts):
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
    train_dataset = PartDataset(npoints=4096, classification=False)
    num = len(train_dataset)
    point_cloud, seg, cls = train_dataset[random.randint(0, num)]
    for k, v in train_dataset.classes.items():
        if cls == v:
            print(k)

    show_point_cloud_part(point_cloud, seg)
    show_point_cloud(point_cloud)
