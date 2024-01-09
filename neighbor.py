import numpy as np
from scipy.integrate import odeint

n=np.ones((5,5))

# 假设 n 是一个二维数组，i 和 j 是索引
i, j = 0, 0  # 以 (0, 0) 为例，你可以根据实际情况更改这些值

# 确保 i 和 j 的值在有效的范围内
def extract_surrounding_points(n, i, j):
    rows, cols = n.shape
    surrounding_points = np.zeros((3, 3), dtype=n.dtype)

    for row_offset in range(-1, 2):
        for col_offset in range(-1, 2):
            # Calculate the new coordinates
            new_i, new_j = i + row_offset, j + col_offset

            # Check if the new coordinates are within bounds
            if 0 <= new_i < rows and 0 <= new_j < cols:
                surrounding_points[row_offset + 1, col_offset + 1] = n[new_i, new_j]
            else:
                # If out of bounds, append 0
                surrounding_points[row_offset + 1, col_offset + 1] = 0

    return np.array(surrounding_points)

i, j = 2, 4  # Example coordinates

matrix_U = extract_surrounding_points(n, i, j)

matrix_B = np.array([[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1, -1]])

# result = np.multiply(result, matrix_B)
# result = np.sum(result)
t=np.linspace(0, 10, 20)

def diff_equation(x,t,matrix_U,matrix_B):
    matrix_B_mul_U = np.multiply(matrix_U, matrix_B)
    B_mul_U = np.sum(matrix_B_mul_U)
    return(- x + B_mul_U -1)


print(matrix_U)
result = odeint(diff_equation, 0, t, args=(matrix_U,matrix_B,))  # x(0) = 0
print(result)