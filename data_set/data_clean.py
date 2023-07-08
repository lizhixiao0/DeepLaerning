import datetime
import time

import numpy as np
import pandas as pd

# # 去除逗号和分号，更换分隔符
# raw_data = pd.read_csv('../data/LD2011_2014.csv', delimiter=';')
# data = raw_data.replace(',', '.', regex=True)
# data.to_csv('./data.csv')
# 加载数据集
raw_data_pd = pd.read_csv('../data/data.csv', delimiter=',')

# 把第一列数据保存一下供以后使用,转换成时间戳格式
data_timestamp = pd.to_datetime(raw_data_pd['time'])
# 将日期时间类型转换为时间戳
timestamp_column = data_timestamp.map(pd.Timestamp.timestamp)

# 格式转换
# 先转成numpy类型
raw_data = raw_data_pd.values

raw_data = raw_data[:, 1:].astype(np.float32)
# 先除以4，让单位变成kw/h 保留4位小数
# raw_data = np.around(c, decimals=4)
raw_data = raw_data / 4
df = pd.DataFrame(raw_data)
df_round=df.round(4)
df_round.to_csv('load_data.csv',index=False,header=False)
timestamp_column.to_csv('timestamp.csv', index=False, header=False)
