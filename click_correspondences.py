'''
  File name: click_correspondences.py
  Author: 
  Date created: 
'''

'''
  File clarification:
    Click correspondences between two images
    - Input im1: target image
    - Input im2: source image
    - Output im1_pts: correspondences coordiantes in the target image
    - Output im2_pts: correspondences coordiantes in the source image
'''
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cpselect import cpselect
def click_correspondences(im1, im2):
    '''
    Tips:
      - use 'matplotlib.pyplot.subplot' to create a figure that shows the source and target image together
      - add arguments in the 'imshow' function for better image view
      - use function 'ginput' and click correspondences in two images in turn
      - please check the 'ginput' function documentation carefully
        + determine the number of correspondences by yourself which is the argument of 'ginput' function
        + when using ginput, left click represents selection, right click represents removing the last click
        + click points in two images in turn and once you finish it, the function is supposed to 
          return a NumPy array contains correspondences position in two images
    '''

    # TODO: Your code here
    # here the im1 is the source img,
    # im2 is the target img.

    # temp = im1[:,:,0]
    # specify the H1,W1 & H2,W2
    size_H, size_W, _ = im1.shape
    # target_H, target_W, _ = im2.shape
    # add boundary points
    left_up = np.array([[0, 0]])
    right_up = np.array([[size_W - 1, 0]])
    left_down = np.array([[0, size_H - 1]])
    right_down = np.array([[size_W - 1, size_H - 1]])
    up_mid = np.array([[np.round((size_W - 1)/2), 0]])
    left_mid = np.array([[0, np.round((size_H - 1)/2)]])
    right_mid = np.array([[size_W - 1, np.round((size_H - 1)/2)]])
    down_mid = np.array([[np.round((size_W - 1)/2), size_H - 1]])



    # add feature points location
    im1_pts, im2_pts = cpselect(im1, im2)
    # add boundary points' location to the feature points' location
    im1_pts = np.vstack((im1_pts, left_up, right_up,left_down, right_down,\
                         up_mid,left_mid,right_mid,down_mid))
    im2_pts = np.vstack((im2_pts, left_up, right_up,left_down, right_down,\
                         up_mid,left_mid,right_mid,down_mid))
    # print(im1_pts)
    # print(im2_pts)
    return im1_pts, im2_pts

if __name__ == '__main__':
    # identify the target and source file
    source_file_name = 'source.jpg'
    target_file_name = 'target.jpg'
    # read in the target & source file
    im1 = np.array(Image.open(source_file_name).convert('RGB'))
    im2 = np.array(Image.open(target_file_name).convert('RGB'))
    # print(im1.shape)


    im1_pts, im2_pts = click_correspondences(im1, im2)
