# -*- coding: utf-8 -*-
"""
Created on Thr Sep 24 11:20:09 2019

@author: M. Farhan Tandia / 0860814
"""
#Import necessary library
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

# #Input the image and get it's information
img = Image.open(sys.argv[1])
img_in = np.array(img)
img_dim= np.size(img_in.shape)

#Make image to 1 channel graysclae image
if(img_dim==3):
    height,width,depth = img_in.shape
    print('Image raw size :\n',height,width,depth)
    #convert 3D image to 2D image
    img_in=img_in[:,:,0]
    height,width = img_in.shape
    print('Image gray size :\n',height,width)
    
else:
    height,width = img_in.shape
    print('Image gray size :\n',height,width)
    

#DFT Formula
def imaginary_e(n):
    return complex(math.cos(n), math.sin(n))

def dft(data):
    n = len(data)
    return [sum((data[j] * imaginary_e(-2 * math.pi * i * j / n) for j in range(n)))
            for i in range(n)]

row = np.transpose(img_in)
fourier_row = np.asarray(dft(row))

column = np.transpose(fourier_row)
fourier_column = np.asarray(dft(column))
    
# compute the common logarithm of each value to reduce the dynamic range
fourier = 20*np.log(np.abs(fourier_column))

# normalize image data 
lowest = np.nanmin(fourier[np.isfinite(fourier)])
highest = np.nanmax(fourier[np.isfinite(fourier)])
data_range = highest - lowest
normalized_fourier = (fourier - lowest) / data_range * 255

#Shift The DC-value (i.e. F(0,0)) to the center
imheight,imwidth=normalized_fourier.shape
height = imheight//2
width = imwidth//2

p1 = normalized_fourier[:int(height),:int(width)]
p2= normalized_fourier[int(height):,:int(width)]
p3= normalized_fourier[:int(height),int(width):]
p4 = normalized_fourier[int(height):,int(width):]
part1 = np.concatenate((p4,p3))
part2 = np.concatenate((p2,p1))
shifted_img = np.concatenate((part1,part2), axis=1)

# convert the normalized data into an image
dft_img_out= Image.fromarray(shifted_img.astype(np.uint8))
print("Image transformed to frequency domain.")
# #image plot
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
# ax1.imshow(img, cmap='gray')
# ax2.imshow(dft_img_out, cmap='gray')
# ax1.title.set_text('Original Image')
# ax2.title.set_text('Fourier Image')
# plt.show()
#save and show the image
dft_img_out.save('0860814.png')
#img_save.show()
