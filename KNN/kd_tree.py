"""
create: 2023/3/19 22:25
author: Yongkui Lai (lykhui@qq.com)

KD-Tree，一种KNN寻找最近邻的方法
核心思想：坐标系法，每次根据一个坐标维度寻找middle数据点（root节点），
        以该数据点按照该坐标维度，将剩下的所有数据点划分为左右两份（left节点和right节点）
"""

import numpy as np
import matplotlib.pyplot as plt
from binarytree import Node


def gen_binary_data(num=3):
    """生成二分类的数据"""
    data_one = np.random.normal([1, 1], [1, 1], [num, 2])
    labels_one = np.ones(num)
    data_two = np.random.normal([0, 0], [1, 1], [num, 2])
    labels_two = np.ones(num)
    mixer_data = np.round(np.concatenate([data_one, data_two]), decimals=2)
    mixer_labels = np.concatenate([labels_one, labels_two])
    # 绘制point
    plt.scatter(data_one[:, 0], data_one[:, 1], c='r')
    plt.scatter(data_two[:, 0], data_two[:, 1], c='g')
    return mixer_data, mixer_labels


def gen_random_test_data():
    """生成随机的测试数据"""
    point = np.random.normal([0.5, 0.5], [1, 1])
    return point


def gen_kd_tree(data, labels, idx, n_dim, min_val, max_val):
    """生成kd-tree"""
    # 如果data中只有1个数据点，直接写入到node中并返回
    if data.shape[0] == 1:
        return Node(','.join([str(i) for i in data[0]]) + '|' + str(labels[0]))
    # 划分的维度
    cur_dim = idx % n_dim
    # 寻找划分维度上的中位点下标
    num = data.shape[0]
    data_indexs = np.arange(num)
    data_indexs = sorted(data_indexs, key=lambda x: data[x][cur_dim])
    mid = num // 2
    mid_idx = data_indexs[mid]
    # 将该中位点写入到node中
    node = Node(','.join([str(i) for i in data[mid_idx]]) + '|' + str(labels[mid_idx]))
    # 计算该维度上的min和max值
    new_min_val = data[data_indexs[0]][cur_dim]
    new_max_val = data[data_indexs[-1]][cur_dim]
    next_dim = (cur_dim + 1) % n_dim
    # 绘图：根据中位点切分平面（只针对2维数据点）
    if n_dim == 2:
        x = np.linspace(min_val, max_val, 256)
        y = np.ones(256) * data[mid_idx][cur_dim]
        if cur_dim == 0:
            x, y = y, x
        plt.plot(x, y)
    # 依据中位点将数据划分为左右两个集合
    left_data, left_labels = data[data_indexs[:mid]], labels[data_indexs[:mid]]
    right_data, right_labels = data[data_indexs[mid + 1:]], labels[data_indexs[mid + 1:]]
    # 遍历左右两个集合
    if left_data.shape[0] > 0:
        node.left = gen_kd_tree(left_data, left_labels, idx + 1, n_dim, new_min_val-1, data[mid_idx][cur_dim])
    if right_data.shape[0] > 0:
        node.right = gen_kd_tree(right_data, right_labels, idx + 1, n_dim, data[mid_idx][cur_dim], new_max_val+1)

    return node


if __name__ == '__main__':
    # 生产有label的“训练”数据
    (trainData, trainLabels) = gen_binary_data()
    # 生成kd-tree
    n_dim = trainData.shape[-1]
    startDim = 0
    nextDim = 1
    num = trainData.shape[0]
    indexs = sorted(np.arange(num), key=lambda x: trainData[x][nextDim])
    leftNum = trainData[indexs[0]][nextDim]
    rightNum = trainData[indexs[-1]][nextDim]
    tree = gen_kd_tree(trainData, trainLabels, startDim, n_dim, leftNum-1, rightNum+1)
    # 打印kd-tree
    print(tree)
    # 数据点划分展示
    plt.show()
