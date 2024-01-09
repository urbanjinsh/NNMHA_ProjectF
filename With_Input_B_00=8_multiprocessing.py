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
import time




# Convert to grayscale
img = Image.open(r"lenna.jpg").convert('L')


# Convert the image to a NumPy array
img = np.array(img)
# print("Shape of the image array:", img.shape) # (height,width)


# normalize input pic to [-1, 1] 
def normalize(im):
    min, max = im.min(), im.max()

    return 2*(im.astype(float)-min)/(max-min) -1 

n=normalize(img)


# A = [[0, 0, 0], 
#       [0,a_00,0], 
#       [0, 0, 0] ]
# B = [[0, 0, 0], 
#       [0,b_00,0], 
#       [0, 0, 0] ]

# ODE 
# a_00 = 3, B_00= 8, z=4
##############################################################
def diff_equation(x,t,u):
    return(- x + 3* (abs(x+1) - abs(x-1)) /2 +8*u + 4)
def nonlinearity (x):          # standard nonlinearity
    return (abs(x+1) - abs(x-1)) /2

t=np.linspace(0, 10, 20)

output = np.zeros(img.shape)

# u_time = 0.00
# result_time = 0.00
# y_time = 0.00

def process_pixel(args):
    i, j = args
    u = n[i][j]
            # end_u_time = time.time()
            # u_time = (end_u_time - start_u_time)+u_time
        
            ## odeint
            # start_result_time = time.time()
    result = odeint(diff_equation, 0, t, args=(u,))  # x(0) = 0
            # result = solve_ivp(diff_equation, [0,10], [0], args=(u,))  # x(0) = 0
            # end_result_time = time.time()
            # result_time = (end_result_time - start_result_time)+result_time

            # start_y_time = time.time()
    y = nonlinearity (result[:, 0])
            # end_y_time = time.time()
            # y_time = (end_y_time - start_y_time)+y_time
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
result.save('result.jpg')



















# a_00 = 2
# x_ij = np.linspace(-13, 13, 100) 
# x_ij_dot = -x_ij + a_00* (abs(x_ij+1) - abs(x_ij-1)) /2 
# plt.plot(x_ij, x_ij_dot)