# -*- coding: utf-8 -*-
"""
Created on Sun Sep  29 13:10:55 2019

@author: M. Farhan Tandia / 0860814
"""
#Import necessary library
from PIL import Image
import numpy as np
import sys

#Initialize cluster dot matrix 
clusterdot = np.array([[62,57,48,36,37,49,58,63],
                        [56,47,35,21,22,38,50,59],
                        [46,34,20,10,11,23,39,51],
                        [33,19,9,3,0,4,12,24],
                        [32,18,8,2,1,5,13,25],
                        [45,31,17,7,6,14,26,40],
                        [55,44,30,16,15,27,41,52],
                        [61,54,43,29,28,42,53,60]])
                
print('Cluster dot screen :\n',clusterdot)
N=len(clusterdot)   

#therholding function
def thresholding (clusterdot):
    screen=(255*(clusterdot+0.5)/N**2)
    return screen

#Input the image and get it's information
img = Image.open(sys.argv[1])
img_in = np.array(img)
print('img_in :\n',img_in)
img_dim=np.size(img_in.shape)
print('Image dim :',img_dim)

#Make image to 2 dimension graysclae image
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

x = width//N
y = height//N

#set a map for thresholding operation to the image
datamap = np.zeros([(y+1)*8,(x+1)*8])
datamap[0:height, 0:width]=img_in.copy()
clusdot = np.tile(clusterdot,(y+1,x+1))
convertmap = thresholding(clusdot)

#compare gray image with threshold image and apply ordered dither algorithm
b = (datamap>=convertmap)
outdata = b[0:height,0:width].copy()
img_out = Image.fromarray((outdata*255).astype(np.uint8))

#save and show the image
img_out.save('0860814.png')
#img_save.show()
