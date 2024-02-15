# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 21:05:45 2023

@author: yanxi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

fig = plt.figure()
x_ij = np.linspace(-13, 13, 100) 



for a_00 in (np.linspace(-10, 10, 10) ):
    x_ij_dot = -x_ij + a_00* (abs(x_ij+1) - abs(x_ij-1)) /2 
    plt.plot(x_ij, x_ij_dot)

plt.axline((-1, -9), slope=-1, color='r', linestyle=(0, (5, 5)))
plt.axline((-1, 11), slope=-1, color='r', linestyle=(0, (5, 5)))
plt.axvline(x=0, color = 'black')
plt.axhline(y=0, color = 'black')
# naming the x axis 
plt.xlabel('x_ij') 
# naming the y axis 
plt.ylabel('x_ij_dot') 




fig = plt.figure()

for z in (np.linspace(-10, 10, 10) ):
    x_ij_dot = -x_ij + a_00* (abs(x_ij+1) - abs(x_ij-1)) /2  + z
    plt.plot(x_ij, x_ij_dot)
    
plt.axline((0, 0), slope=-1, color='black', linestyle=(0, (5, 5)))
plt.axvline(x=0, color = 'black')
plt.axhline(y=0, color = 'black')
# naming the x axis 
plt.xlabel('x_ij') 
# naming the y axis 
plt.ylabel('x_ij_dot') 

plt.show()