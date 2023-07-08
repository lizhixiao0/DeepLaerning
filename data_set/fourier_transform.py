import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft

load_data = np.loadtxt('../data/load_data.csv', delimiter=',')
# 先分析一个用户的负荷信息
load_0 = load_data[:, 3]
plt.plot(load_0)
plt.show()

time_interval = 0.25
# 执行FFT计算
spectrum = fft(load_0)
# 获取振幅谱
amplitude_spectrum = np.abs(spectrum)

# 采样频率和信号长度
sample_rate = 4 / 60 / 60
signal_length = len(load_0)

# 频率轴
frequency_axis = np.fft.fftfreq(signal_length, d=1 / sample_rate)

# 绘制频谱图

# 设置阈值，用于筛选极大值
threshold = 20000  # 根据实际情况进行调整
max_freq = []
# 获取满足阈值条件的极大值索引
max_indices = np.argwhere(amplitude_spectrum > threshold).flatten()
# 获取对应的频率值
max_frequencies = frequency_axis[max_indices]

positive_freq_indices = frequency_axis > 0
plt.plot(frequency_axis[positive_freq_indices], amplitude_spectrum[positive_freq_indices])

plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum')
plt.xlim(0, 0.00006)
plt.grid(True)

# 添加文本标签显示极大值的频率坐标
for frequency, amplitude in zip(max_frequencies, amplitude_spectrum[max_indices]):
    plt.annotate(f'{frequency:.8f}', xy=(frequency, amplitude), xytext=(frequency, amplitude + 10),
                 arrowprops=dict(facecolor='black', arrowstyle='-'))
    max_freq.append(frequency)
plt.show()

t = (1 / np.array(max_freq)) / 60 / 60 / 24
print(t)
