# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:56:19 2023

@author: Bingyi Jin
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from sympy import solve
from sympy import *
from multiprocessing import Pool

################################################################
# set mean value、variance and the noise mode

mean_value = 0
variance_value = 0.001
mode = 1 #1 stand for varing noise, 0 stand for consitent noise
################################################################


# Convert to grayscale
img = Image.open(r'pic/black_and_white.jpg').convert('1')


# Convert the image to a NumPy array
img = np.array(img)
# print("Shape of the image array:", img.shape) # (height,width)

img = img.astype(float)
# normalize input pic to [-1, 1] 
def normalize(im):
    min, max = im.min(), im.max()

    return 2 * (im.astype(float)-min)/(max-min) - 1

n=normalize(img)
# n =img
# result_n = Image.fromarray(n.astype(np.uint8))
# plt.imshow(n)
# result_n.save('pic/result_n.jpg')

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

def generate_gaussian_noise_matrix(mean, variance):
    std_dev = np.sqrt(variance)
    noise_matrix = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            noise_matrix[i, j] = np.random.normal(mean, std_dev)

    return noise_matrix

def add_noise(noise_matrix, matrix):
    return noise_matrix + matrix

# A = [[0, 0, 0], 
#       [0,a_00,0], 
#       [0, 0, 0] ]
# B = [[0, 0, 0], 
#       [0,b_00,0], 
#       [0, 0, 0] ]


##############################################################
a_00 = 2

matrix_B = np.array([[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1, -1]])

def diff_equation(x,t,a_00,matrix_U,matrix_B):
    matrix_B_mul_U = np.multiply(matrix_U, matrix_B)
    B_mul_U = np.sum(matrix_B_mul_U)
    return(- x + a_00 * nonlinearity(x) + B_mul_U - 0.5)
def nonlinearity(x):          # standard nonlinearity
    return (abs(x+1) - abs(x-1)) /2

t=np.linspace(0, 5, 30)
# t=np.linspace(0, 30, 60)

output = np.zeros(img.shape)



consitent_noise = generate_gaussian_noise_matrix(mean_value, variance_value)

def process_pixel(args):
    i, j = args
    varing_noise = generate_gaussian_noise_matrix(mean_value, variance_value)
    matrix_U = extract_surrounding_points(n, i, j)
    if mode == 1:   
        matrix_U = add_noise(varing_noise,matrix_U)

    elif mode == 0:
        matrix_U = add_noise(consitent_noise,matrix_U)
        
    result = odeint(diff_equation, -0.01, t, args=(a_00,matrix_U,matrix_B,))  # x(0) = 0

    y = nonlinearity (result[:, 0])

    # print(y)
    return i,j,y



if __name__ == '__main__':
    with Pool() as pool:
        
        args_list = [(i, j) for i in range(img.shape[0]) for j in range(img.shape[1])]

        
        results = pool.map(process_pixel, args_list)

    for i, j, y in results:
        output[i][j] = y[-1]

    print(output.shape)
        

result = Image.fromarray(output.astype(np.uint8))
plt.imshow(result)
result.save('pic/result_with_varience_noise_0.005.jpg')



















# a_00 = 2
# x_ij = np.linspace(-13, 13, 100) 
# x_ij_dot = -x_ij + a_00* (abs(x_ij+1) - abs(x_ij-1)) /2 
# plt.plot(x_ij, x_ij_dot)