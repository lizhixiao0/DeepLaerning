import numpy as np


def zscore_normalize(data):
    """
    z-score标准化
    输入数据集，返回每列标准化的数据集。
    :return: 标准化后的数据，mean，std
    """
    mean = np.mean(data)
    std = np.std(data)
    normalize_data = (data - mean) / std
    return normalize_data, mean, std


def zscore_denormalize(normalized_data, mean, std):
    """
    z-score标准化还原
    """
    denormalized_data = normalized_data * std + mean
    return denormalized_data


