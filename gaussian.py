import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_noise(mean, variance, size):
    """
    生成高斯白噪声
    
    参数：
    - mean: 均值
    - variance: 方差
    - size: 噪声序列的大小
    
    返回：
    - 高斯白噪声序列
    """
    std_dev = np.sqrt(variance)
    noise = np.random.normal(mean, std_dev, size)
    return noise

def generate_gaussian_noise_matrix(mean, variance):
    std_dev = np.sqrt(variance)
    noise_matrix = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            noise_matrix[i, j] = np.random.normal(mean, std_dev)

    return noise_matrix
# 设置均值、方差和生成噪声的大小
mean_value = 0
variance_value = 0.001
noise_size = 100

# 生成高斯白噪声
gaussian_noise = generate_gaussian_noise(mean_value, variance_value, noise_size)
gaussian_noise_matrix = generate_gaussian_noise_matrix(mean_value, variance_value)

# 打印前10个值
# print(gaussian_noise[:10])

# 绘制高斯白噪声图形
# plt.plot(gaussian_noise)
# plt.title('Gaussian White Noise')
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.show()

print(gaussian_noise_matrix)