# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:17:19 2023

@author: yanxia
"""
#BGR order if you used cv2. imread()#
# PIL follows RGB color convention
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import time





#image = Image.open(image_path).convert('L')  # Convert to grayscale
img = Image.open(r"test.jpg")
#resize 
# img = img.resize((600, 400))
# img.save('600x400.jpg')

img=img.convert('L')
# plt.imshow(img)

#img_color = Image.merge('RGB', [img]*3)
#img_color.save('new.png')



# Convert the image to a NumPy array
img = np.array(img)
print("Shape of the image array:", img.shape) # (height,width)


# normalize input pic to [-1, 1] 
def normalize(im):
    min, max = im.min(), im.max()
    #print (min, max)
    return 2*(im.astype(float)-min)/(max-min) -1 
# print (normalize (img))
n=normalize(img)
# result = Image.fromarray(n.astype(np.uint8))
# result.save('result.png')


# A = [[0, 0, 0], 
#       [0,a_00,0], 
#       [0, 0, 0] ]
# B = [[0, 0, 0], 
#       [0,b_00,0], 
#       [0, 0, 0] ]


# a_00 = 3, B_00= 8, z=4
##############################################################
def diff_equation(x,t,u):
    return(- x + 3* (abs(x+1) - abs(x-1)) /2 +8*u + 4)
def nonlinearity (x):          # standard nonlinearity
    return (abs(x+1) - abs(x-1)) /2


start = time.time()


output = np.zeros(img.shape )
for i in range (0,img.shape[0]):
    for j in range (0,img.shape[1]):
        # print (img_norm[i][j])
        u = n[i][j]
        ## solve_ivp
        # F=lambda t,x: - x + 3* (abs(x+1) - abs(x-1)) /2 -2*u + 4
        # t_eval = np.linspace(0, 20, 200)
        # sol = solve_ivp(F, [0, 20], [0], t_eval=t_eval)
        # plt.plot(sol.t, sol.y[0])
        
        ## odeint
        ## from plt.plot(t, result[:, 0], label='x(t)') we see that the curves already converge before 10. 
        t=np.linspace(0, 10, 20)
        result = odeint(diff_equation, 0, t, args=(u,))  # x(0) = 0
        # plt.plot(t, result[:, 0], label='x(t)')
        
        # # calculate all y(t) 
        # y = nonlinearity (result[:, 0]) 
        # # plt.plot(t, y, label='y(t)')
        # output[i][j]=y[-1]  
        
        y_endtime = nonlinearity (result[:, 0][-1]) # only takes x[20] 
        # plt.plot(t, y, label='y(t)')
        output[i][j]=y_endtime
# plt.grid()  
# plt.show() 

end = time.time()
print(f'elapsed time: { (end - start)/60:0.2f} min')
result_test = Image.fromarray(output.astype(np.uint8))
plt.imshow(result_test)
result_test.save('result_test.jpg')



















# a_00 = 2
# x_ij = np.linspace(-13, 13, 100) 
# x_ij_dot = -x_ij + a_00* (abs(x_ij+1) - abs(x_ij-1)) /2 
# plt.plot(x_ij, x_ij_dot)