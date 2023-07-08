import argparse
import pickle

import matplotlib.pyplot as plt
import torch
from torch import nn

from load_forecast.data_load import DataLoad
from load_forecast.model.test_lstm import LSTMModel
from load_forecast.model.test_rnn import RNNModel
from load_forecast.utils import evaluate_predictions

parser = argparse.ArgumentParser(description=' All params')
# ----dataset
parser.add_argument('--raw_data_src', type=str,
                    default=r'E:\lin\DeepLearning\LiNet\data_set\load_data.csv',
                    help='location of the data file')  # required=True,
parser.add_argument('--de_noise_date_src', type=str,
                    default=r'E:\lin\DeepLearning\LiNet\data_set\De_noise_data_01_32.csv',
                    help='location of the data file')  # required=True,
parser.add_argument('--horizon', type=int, default=1, metavar='N', help='horizon')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size (default: 32)')
parser.add_argument('--window', type=int, default=7 * 4 * 24, metavar='N', help='window_step')
parser.add_argument('--window_step', type=int, default=1, metavar='N', help='window_step')
parser.add_argument('--feature_size', type=int, default=5, metavar='N', help='feature size')
parser.add_argument('--load_index', type=int, default=0, metavar='N', help='load_index')
parser.add_argument('--cuda', type=bool, default=True, help='If use cuda')
# para = parser.parse_args()
para = parser.parse_known_args()[0]

#
# 加载数据集
dataset = DataLoad(para)
# # 保存数据集
# with open('dataset.pkl', 'wb') as f:
#     pickle.dump(dataset, f)
#
# # 加载数据集
# with open('dataset.pkl', 'rb') as f:
#     dataset = pickle.load(f)

# 设置LSTM模型参数
input_size = 1  # 输入特征维度
hidden_size = 64  # LSTM隐藏层大小
output_size = 1  # 输出维度

# 创建模型实例
model = LSTMModel(input_size, hidden_size, output_size)
# model = RNNModel(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# 检测显卡是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用第一个可用的显卡
    print("显卡可用，使用显卡进行训练")
else:
    device = torch.device("cpu")
    print("未检测到显卡，使用CPU进行训练")

# 设置模型和数据集的设备
model.to(device)
criterion.to(device)

# 训练模型
print('---start train---')
batch_size = para.batch_size
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for X, Y in dataset.get_batches(dataset.train_data, dataset.train_labels, batch_size=batch_size, shuffle=True,
                                    cuda=True):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
    print(f'epoch={epoch + 1}')
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 在验证集上进行预测
print('---valid---')
model.eval()
val_outputs_list = []  # 存储每次的val_outputs值
val_real_list = []  # 存储每次的val_outputs值
with torch.no_grad():
    for X, Y in dataset.get_batches(dataset.val_data, dataset.val_labels, batch_size=batch_size, shuffle=False,
                                    cuda=True):
        val_outputs = model(X)
        val_loss = criterion(val_outputs, Y)
        val_outputs_list.append(val_outputs)  # 将val_outputs添加到列表中
        val_real_list.append(Y)
    print(f"Validation Loss: {val_loss.item():.4f}")

# 将列表转换为张量
val_outputs_tensor = torch.cat(val_outputs_list, dim=0)
val_real_tensor = torch.cat(val_real_list, dim=0)
val_real_data = dataset.inverse_normalize_data(val_real_tensor)
val_predict_data = dataset.inverse_normalize_data(val_outputs_tensor)
print('Valid---rmse=%.4f, mse=%.4f, mae=%.4f' % evaluate_predictions(val_predict_data, val_real_data))

# 在测试集上进行预测
print('---test---')
model.eval()
test_predict_list = []
test_real_list = []
with torch.no_grad():
    for X, Y in dataset.get_batches(dataset.test_data, dataset.test_labels, batch_size=batch_size, shuffle=False,
                                    cuda=True):
        test_outputs = model(X)
        test_loss = criterion(test_outputs, Y)
        test_predict_list.append(test_outputs)
        test_real_list.append(Y)
    print(f"Test Loss: {test_loss.item():.4f}")

test_outputs_tensor = torch.cat(test_predict_list)
test_real_tensor = torch.cat(test_real_list)

test_predict_data = dataset.inverse_normalize_data(test_outputs_tensor)
test_real_data = dataset.inverse_normalize_data(test_real_tensor)
print('rmse=%.4f, mse=%.4f, mae=%.4f' % evaluate_predictions(test_predict_data, test_real_data))

real_data=dataset.raw_test_y[:test_predict_data.size]
# 创建图形对象
fig = plt.Figure()
# 绘制数据
plt.plot(test_real_data[-4 * 24:], 'r-', label='Real Data')
plt.plot(test_predict_data[-4 * 24:], 'b:', label='Predicted Data')
plt.plot(real_data[-4 * 24:],label='True Real Data')
plt.plot()
# 添加图例
plt.legend()
# 添加标题
plt.title('Comparison of Real and Predicted Data')
# 显示图形
plt.show()
