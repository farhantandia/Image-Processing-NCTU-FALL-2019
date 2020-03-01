from PIL import Image
import numpy as np
import sys

#Make our own structuring element
#I use 3x3 cross SE for my testing
'''
cross = np.array([[0,255,0],
                        [255,255,255],
                        [0,255,0]],np.uint8)
cross[cross<127] = 0
cross[cross>127] = 1
img = Image.fromarray(cross)
img.show()
img.save('se.png')
'''

#Input the image and get it's information
img = Image.open(sys.argv[1])
se = Image.open(sys.argv[2])
ste_origin = int(sys.argv[3])
img_in = np.array(img)
img_se = np.array(se)

#Origin coordinate cannot more than Structure element shape
if ste_origin > img_se.shape[0]:
    raise Exception("Origin coordinate beyond the structuring element limit")

print("origin point:",(ste_origin,ste_origin))

img_dim=np.size(img_in.shape)

#Make image to 2 channel graysclae image
if(img_dim==3):
    
    #convert 3D image to 2D image
    img_in=img_in[:,:,0]
    height,width = img_in.shape
    print('Image gray size :\n',height,width)
    
else:
    height,width = img_in.shape
    print('Image gray size :\n',height,width)

print('Input Image Shape :\n',img_in.shape)
print('Structure element :\n', img_se)

#Convert to Binary Image
img_in[img_in<127] = 0
img_in[img_in>127] = 1

img_se[img_se<127] = 0
img_se[img_se>127] = 1


def idx_check(index):
    if index < 0:
        return 0
    else:
        return index

def dilation(binary_img_matrix , structuring_element):
    binary_img_matrix = np.asarray(binary_img_matrix)
    structuring_element = np.asarray(structuring_element)
    ste_shp = structuring_element.shape
    dilated_img = np.zeros((binary_img_matrix.shape[0], binary_img_matrix.shape[1]))
#     ste_origin = ((structuring_element.shape[0]-1)/2, (structuring_element.shape[1]-1)/2)
    ste_origin = (int(np.ceil((structuring_element.shape[0] - 1) / 2.0)), int(np.ceil((structuring_element.shape[1] - 1) / 2.0)))
    for i in range(len(binary_img_matrix)):
        for j in range(len(binary_img_matrix[0])):
            overlap = binary_img_matrix[idx_check(i - ste_origin[0]):i + (ste_shp[0] - ste_origin[0]), 
                    idx_check(j - ste_origin[1]):j + (ste_shp[1] - ste_origin[1])]
            shp = overlap.shape

            ste_first_row_idx = int(np.fabs(i - ste_origin[0])) if i - ste_origin[0] < 0 else 0
            ste_first_col_idx = int(np.fabs(j - ste_origin[1])) if j - ste_origin[1] < 0 else 0

            ste_last_row_idx = ste_shp[0] - 1 - (i + (ste_shp[0] - ste_origin[0]) - binary_img_matrix.shape[0]) if i + (ste_shp[0] - ste_origin[0]) > binary_img_matrix.shape[0] else ste_shp[0]-1
            ste_last_col_idx = ste_shp[1] - 1 - (j + (ste_shp[1] - ste_origin[1]) - binary_img_matrix.shape[1]) if j + (ste_shp[1] - ste_origin[1]) > binary_img_matrix.shape[1] else ste_shp[1]-1

            if shp[0] != 0 and shp[1] != 0 and np.logical_and(structuring_element[ste_first_row_idx:ste_last_row_idx+1, ste_first_col_idx:ste_last_col_idx+1], overlap).any():
                dilated_img[i, j] = 1
    return dilated_img

def erosion(binary_img_matrix, structuring_element):
    binary_img_matrix = np.asarray(binary_img_matrix)
    structuring_element = np.asarray(structuring_element)
    ste_shp = structuring_element.shape
    eroded_img = np.zeros((binary_img_matrix.shape[0], binary_img_matrix.shape[1]))
    ste_origin = (int(np.ceil((structuring_element.shape[0] - 1) / 2.0)), int(np.ceil((structuring_element.shape[1] - 1) / 2.0)))
    for i in range(len(binary_img_matrix)):
        for j in range(len(binary_img_matrix[0])):
            overlap = binary_img_matrix[idx_check(i - ste_origin[0]):i + (ste_shp[0] - ste_origin[0]),
                      idx_check(j - ste_origin[1]):j + (ste_shp[1] - ste_origin[1])]
            shp = overlap.shape
            ste_first_row_idx = int(np.fabs(i - ste_origin[0])) if i - ste_origin[0] < 0 else 0
            ste_first_col_idx = int(np.fabs(j - ste_origin[1])) if j - ste_origin[1] < 0 else 0

            ste_last_row_idx = ste_shp[0] - 1 - (i + (ste_shp[0] - ste_origin[0]) - binary_img_matrix.shape[0]) if i + (ste_shp[0] - ste_origin[0]) > binary_img_matrix.shape[0] else ste_shp[0]-1
            ste_last_col_idx = ste_shp[1] - 1 - (j + (ste_shp[1] - ste_origin[1]) - binary_img_matrix.shape[1]) if j + (ste_shp[1] - ste_origin[1]) > binary_img_matrix.shape[1] else ste_shp[1]-1

            if shp[0] != 0 and shp[1] != 0 and np.array_equal(np.logical_and(overlap, structuring_element[ste_first_row_idx:ste_last_row_idx+1,
                                                                       ste_first_col_idx:ste_last_col_idx+1]),structuring_element[ste_first_row_idx:ste_last_row_idx+1,
                                                                       ste_first_col_idx:ste_last_col_idx+1]):
                eroded_img[i, j] = 1
    return eroded_img

dilated_img = dilation(img_in,img_se)
eroded_img = erosion(img_in,img_se)

#Convert Binary Image (0 and 1) to Image .png (0-255)
dilated_img[dilated_img==1] = 255
new_dilated_img = Image.fromarray(np.asarray(dilated_img,np.uint8))
eroded_img[eroded_img==1] = 255
new_eroded_img =Image.fromarray(np.asarray(eroded_img,np.uint8))

new_dilated_img.save('08060814_dilation.png')
new_eroded_img.save('08060814_erosion.png')