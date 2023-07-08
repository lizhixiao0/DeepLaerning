import random

import numpy as np
import torch


class DataLoad(object):
    """
    数据加载类，用于加载、标准化、划分数据集等操作。
    Args:
        para (object): 参数对象，包含窗口大小、预测步长、窗口跳跃步长、负荷特征索引和数据集路径。
    """

    def __init__(self, para):
        self.window = para.window  # 时间窗口
        self.horizon = para.horizon  # 预测步长
        self.window_step = para.window_step  # 窗口跳跃步长
        self.load_idx = para.load_index  # 负荷特征所在的列
        self.raw_data = None
        self.raw_data = self._data_load(para.raw_data_src) # 加载原始数据
        self.de_noise_data = self._data_load(para.de_noise_date_src)    # 加载去噪后的数据
        self.data, self.mean, self.std = self._normalize_data(self.de_noise_data)
        self.window_data, self.y_index = self._split_window(self.data, self.window, self.horizon, self.window_step,
                                                            self.load_idx)  # 获得窗口化的数据

        (self.train_data, self.train_labels), (self.val_data, self.val_labels), (
            self.test_data, self.test_labels), (
            self.raw_train_y, self.raw_val_y, self.raw_test_y) = self._split_dataset(self.window_data, 0.8, 0.1, 0.1,
                                                                                     shuffle=False)

    @staticmethod
    def _data_load(src):
        """
        :param src: 数据集路径
        :return: 加载数据集到self.raw_data
        """
        raw_data = np.loadtxt(src, delimiter=',')

        # return raw_data[0:4 * 24 * 365]
        return raw_data[0:4 * 24 * 365, 1:2]

    @staticmethod
    def _normalize_data(data):
        """
        数据标准化 z-标准化
        """
        # data = self.raw_data
        m, n = data.shape
        standardized_data = np.zeros((m, n))
        mean = np.ones(n)
        std = np.ones(n)
        for i in range(n):
            mean[i] = np.mean(data[:, i])
            std[i] = np.std(data[:, i])
            standardized_data[:, i] = (data[:, i] - mean[i]) / std[i]
        return standardized_data, mean, std

    def inverse_normalize_data(self, standardized_data):
        """
        标准化还原数据
        :param standardized_data: 标准化后的数据
        :return: 还原后的原始数据
        """
        if isinstance(standardized_data, torch.Tensor):
            standardized_data = standardized_data.cpu().detach().numpy()
        m, n = standardized_data.shape
        mean = self.mean
        std = self.std
        data = np.zeros((m, n))
        for i in range(n):
            data[:, i] = standardized_data[:, i] * std[i] + mean[i]
        return data

    @staticmethod
    def _split_window(data, window, horizon, window_step, load_idx):
        """
        划分时间窗口
        :param data: 一个数据集,训练集或测试，验证集的数据
        :param window: 时间窗口大小
        :param horizon: 预测时间的步数，未来1个时刻还是2个时刻
        :param window_step: 时间窗口创建时，每次滑动的尺度。
        :param load_idx: 负荷数据特征所在的列
        :return: 滑动时间窗口构建的数据集。和新数据集的labels对应与原始数据集labels的索引。

        ---

        2023-04-25  在生成时间窗口时，为每个时刻的数据添加一个时间戳的属性,放在最后一列
        """
        X = []
        Y = []
        y_index = []
        row, col = data.shape
        # timestamp = np.arange(window).reshape(-1, 1)
        start_idx = 0
        while start_idx + window + horizon < row:
            end_idx = start_idx + window
            predict_idx = end_idx + horizon
            X.append(data[start_idx:end_idx, load_idx:load_idx + 1])
            Y.append(data[end_idx:predict_idx, load_idx])
            start_idx += window_step
            y_index.append(end_idx)
        X = np.asarray(X)  # (n,336,5)
        Y = np.asarray(Y)
        y_index = np.asarray(y_index)
        # 为x添加时间戳的特征
        # timestamp = np.broadcast_to(timestamp, (X.shape[0], X.shape[1], 1))
        # X = np.concatenate([X, timestamp], axis=2)
        return [X, Y], y_index

    def _split_dataset(self, data, train_ratio, test_ratio, val_ratio, shuffle=False):
        """
        划分数据集为训练集、测试集和验证集
        :param data: 原始数据集，已经进行时间窗口划分
        :param train_ratio: 训练集比例
        :param test_ratio: 测试集比例
        :param val_ratio: 验证集比例
        :param shuffle: 是否打乱数据集的顺序
        :return: 划分后的训练集、测试集和验证集数据。和对应的原始数据的labels
        """

        # 获取数据集大小
        dataset_size = len(data[0])

        # 计算划分的数据个数
        train_size = int(train_ratio * dataset_size)
        test_size = int(test_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)

        # 打乱数据集的顺序,在想出更好的办法前，暂时不用他
        if shuffle:
            shuffled_indices = list(range(dataset_size))
            random.shuffle(shuffled_indices)
            data = (data[0][shuffled_indices], data[1][shuffled_indices])

        # 划分数据集
        train_data = data[0][:train_size]
        train_labels = data[1][:train_size]

        val_data = data[0][train_size:train_size + val_size]
        val_labels = data[1][train_size:train_size + val_size]

        test_data = data[0][train_size + val_size:train_size + test_size + val_size]
        test_labels = data[1][train_size + val_size:train_size + test_size + val_size]

        # 获取训练集，验证集，测试集的对应原始数据的索引
        raw_train_labels = self.y_index[:train_size]
        raw_val_labels = self.y_index[train_size:train_size + val_size]
        raw_test_labels = self.y_index[train_size + val_size:train_size + test_size + val_size]

        # 根据索引获取对应的原始数据的labels
        raw_train_y = self.raw_data[raw_train_labels]
        raw_val_y = self.raw_data[raw_val_labels]
        raw_test_y = self.raw_data[raw_test_labels]

        return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels), (raw_train_y, raw_val_y,
                                                                                              raw_test_y)

    @staticmethod
    def get_batches(inputs, targets, batch_size, shuffle=False, cuda=True):
        """
        获取batch数据的函数，类似DataLoader里封装的函数，这里不过是自己手写的。
        :param inputs: X 数据集的输入
        :param targets: Y 数据集的labels
        :param batch_size: batch_size 大小
        :param shuffle: True用于打乱数据集，每次都会以不同的顺序返回
        :param cuda: 是否使用cuda显卡
        :return: 返回batch块大小的数据
        """
        inputs = torch.from_numpy(inputs).to(torch.float32)
        targets = torch.from_numpy(targets).to(torch.float32)
        length = len(inputs)  # 输入集的长度
        if shuffle:
            index = torch.randperm(length)  # torch.randperm(n)：将0~n-1（包括0和n-1）随机打乱后获得的数字序列，函数名是random permutation缩写
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length - batch_size:
            end_idx = min(length, start_idx + batch_size)  # 取一个batch块的数据，到最后不够了就取到最后一个值
            excerpt = index[start_idx:end_idx]  # 一个batch大小的数据
            X = inputs[excerpt]
            Y = targets[excerpt]
            if cuda:
                X = X.cuda()
                Y = Y.cuda()
            yield X, Y
            start_idx += batch_size
