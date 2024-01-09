


import numpy as np

# 创建两个矩阵
matrix1 = np.array([[1, 2,1], [3, 4,2],[2,6,4]])
matrix2 = np.array([[5, 6], [7, 8]])

def normalize(im):
    min, max = im.min(), im.max()

    return 2 * (im.astype(float)-min)/(max-min) - 1

norm_mat = normalize(matrix1)

t=np.linspace(0, 30, 30)


def nonlinearity (x):          # standard nonlinearity
    return (abs(x+1) - abs(x-1)) /2


print(matrix1)
print(norm_mat)
print(nonlinearity(norm_mat))
