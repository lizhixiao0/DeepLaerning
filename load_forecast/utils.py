import numpy as np
import torch

def evaluate_predictions(predictions, targets):
    """
    计算预测结果与真实值之间的评估指标：RMSE、MSE、MAE

    Args:
        predictions: 预测结果张量
        targets: 真实值张量

    Returns:
        rmse: 均方根误差
        mse: 均方误差
        mae: 平均绝对误差
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    return rmse, mse, mae
