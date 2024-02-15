# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:56:19 2023

This file is to do the edge detection in the CellNN network with noise.
The noise can be set to be consitent or varing.
The edge detection process is done in parallel.

Input: a black and white picture
Output: a picture with edge detection

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
variance_value = 0.005
mode = 1 #1 stand for varing noise, 0 stand for consitent noise

# set the pic path
pic_path = 'pic/lenna_BW.jpg'
################################################################

# Convert to balck and white
img = Image.open(pic_path).convert('1')

# Convert the image to a NumPy array
img = np.array(img)
img = img.astype(float)

# normalize input pic to [-1, 1] 
def normalize(im):
    min, max = im.min(), im.max()

    return 2 * (im.astype(float)-min)/(max-min) - 1

n=normalize(img)

# Extract the surrounding 3 * 3 points of a pixel in (i,j) coordinates
def extract_surrounding_points(n, i, j):
    """
    Generate 3 * 3 neighbours of a pixel which in (i,j) coordinates
    
    parameter:
    - n: target picture
    - i,j: coordinates of the pixel
    
    return: 
    - 3 * 3 matrix
    """
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

# Generate a 3 * 3 matrix of Gaussian noise
def generate_gaussian_noise_matrix(mean, variance):
    """
    Generate a 3 * 3 matrix of Gaussian noise maxtrix
    
    parameter:
    - mean: mean of Gaussian noise
    - variance: variance of Gaussian noise
    
    return: 
    - 3 * 3 matrix of Gaussian noise
    """
    std_dev = np.sqrt(variance)
    noise_matrix = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            noise_matrix[i, j] = np.random.normal(mean, std_dev)

    return noise_matrix

# Add noise to a matrix
def add_noise(noise_matrix, matrix):
    return noise_matrix + matrix


##############################################################
# Define edge detection template
a_00 = 2
matrix_B = np.array([[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1, -1]])

# Define the differential equation
def diff_equation(x,t,a_00,matrix_U,matrix_B):
    matrix_B_mul_U = np.multiply(matrix_U, matrix_B)
    B_mul_U = np.sum(matrix_B_mul_U)
    return(- x + a_00 * nonlinearity(x) + B_mul_U - 0.5)

# Define the nonlinearity
def nonlinearity(x):          # standard nonlinearity
    return (abs(x+1) - abs(x-1)) /2

# Define the time points
t=np.linspace(0, 5, 30)

# Define the output matrix
output = np.zeros(img.shape)

# Define the noise matrix which will not change during the process
consitent_noise = generate_gaussian_noise_matrix(mean_value, variance_value)

# Define the edge detection process of each pixel
def process_pixel(args):
    i, j = args
    varing_noise = generate_gaussian_noise_matrix(mean_value, variance_value) # Define the noise matrix which will change during the process
    matrix_U = extract_surrounding_points(n, i, j)

    # Select the mode to be used
    if mode == 1:   
        matrix_U = add_noise(varing_noise,matrix_U)
    elif mode == 0:
        matrix_U = add_noise(consitent_noise,matrix_U)
        
    result = odeint(diff_equation, -0.01, t, args=(a_00,matrix_U,matrix_B,))  # x(0) = 0

    y = nonlinearity (result[:, 0])

    return i,j,y



if __name__ == '__main__':
    # Do the edge detection in parallel
    with Pool() as pool:
        args_list = [(i, j) for i in range(img.shape[0]) for j in range(img.shape[1])]
        results = pool.map(process_pixel, args_list)
    for i, j, y in results:
        output[i][j] = y[-1]

    print(output.shape)
        
# Save results
result = Image.fromarray(output.astype(np.uint8))
plt.imshow(result)
result.save('pic/lenna_with_varience_noise_{variance}.jpg')
