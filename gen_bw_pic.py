from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 打开彩色图像
img = Image.open(r'pic/lenna.jpg')

# 转换为NumPy数组
img_array = np.array(img)

# 设置阈值，将亮度大于阈值的像素设为白色，否则设为黑色
threshold = 128
bw_img_array = (img_array.mean(axis=2) > threshold).astype(np.uint8) * 255

# 创建新的黑白图像
bw_img = Image.fromarray(bw_img_array)

# 保存黑白图像
bw_img.save('pic/lenna_BW.jpg')
