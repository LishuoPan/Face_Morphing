'''
  File name: morph_tri.py
  Author:
  Date created:
'''

'''
    File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

from scipy.spatial import Delaunay
import numpy as np
import numpy.linalg as LA
from PIL import Image
import scipy
from helpers import interp2
from click_correspondences import click_correspondences
# from images2gif import writeGif


def matrixA_for_three_points(fea_coor, three_points_set):

    A = np.zeros((3, 3))
    # fill in the first two rows using Ax,Ay,Bx,By,Cx,Cy
    for i, element in enumerate(three_points_set):
        A[0:2, i] = fea_coor[element, :]
    # add 1 row in the last row
    A[2, :] = 1
    # print(A)
    return A


def generate_warp(size_H, size_W,
                  Tri, A_Inter_inv_set,
                  A_im_set, image):
    # generate x,y meshgrid
    x, y = np.meshgrid(np.arange(size_W), np.arange(size_H))
    x = x.flatten()
    y = y.flatten()
    # all points in img (size_H*size_W, 2) as x,y system
    empty_points = np.array(list(zip(x, y)))
    # print(empty_points)
    assert empty_points.shape == (size_H * size_W, 2)
    # find the tris where these points live in
    find_simplex_tris = Tri.find_simplex(empty_points)
    # compute alpha, beta, gamma
    all_Inter_inv_A = A_Inter_inv_set[find_simplex_tris]
    all_img_A = A_im_set[find_simplex_tris]

    # aug_empty_points = np.hstack((empty_points,np.ones((empty_points.shape[0], 1))))
    alpha = all_Inter_inv_A[:, 0, 0] * empty_points[:, 0].flatten() + \
        all_Inter_inv_A[:, 0, 1] * empty_points[:, 1].flatten() + \
        all_Inter_inv_A[:, 0, 2] * 1

    beta = all_Inter_inv_A[:, 1, 0] * empty_points[:, 0] + \
        all_Inter_inv_A[:, 1, 1] * empty_points[:, 1] + \
        all_Inter_inv_A[:, 1, 2] * 1

    gamma = all_Inter_inv_A[:, 2, 0] * empty_points[:, 0] + \
        all_Inter_inv_A[:, 2, 1] * empty_points[:, 1] + \
        all_Inter_inv_A[:, 2, 2] * 1

    assert beta.size == empty_points[:, 0].flatten().size
    # print(all_Inter_inv_A[:, 2, 0].shape)
    all_x_coor = all_img_A[:, 0, 0] * alpha + \
        all_img_A[:, 0, 1] * beta + \
        all_img_A[:, 0, 2] * gamma

    all_y_coor = all_img_A[:, 1, 0] * alpha + \
        all_img_A[:, 1, 1] * beta + \
        all_img_A[:, 1, 2] * gamma

    # analytical result
    all_z_coor = np.ones(beta.size)

    all_x_coor_regularized = all_x_coor / all_z_coor
    all_y_coor_regularized = all_y_coor / all_z_coor

    # generate warp img
    generated_pic = np.zeros((size_H, size_W, 3), dtype=np.uint8)

    generated_pic[:, :, 0] = np.reshape(interp2(image[:, :, 0], all_x_coor_regularized, all_y_coor_regularized),
                                        [size_H, size_W])
    generated_pic[:, :, 1] = np.reshape(interp2(image[:, :, 1], all_x_coor_regularized, all_y_coor_regularized),
                                        [size_H, size_W])
    generated_pic[:, :, 2] = np.reshape(interp2(image[:, :, 2], all_x_coor_regularized, all_y_coor_regularized),
                                        [size_H, size_W])

    # LOOP VERSION: SLOW
    #
    #
    # # generate a empty matrix to store the warp img
    # generated_pic = np.zeros((size_H, size_W, 3), dtype=np.uint8)
    #
    # # go through every point's coor in the inter-generated-pic
    # for i in np.arange(size_H):
    #       for j in np.arange(size_W):
    #             # find its belonging triangle,
    #             # find_simplex according to x,y system
    #             fea_index = Tri.find_simplex(np.array([i, j]))
    #             # print(fea_index)
    #             # pull out the A_inter_inv matrix of Tri index
    #             A_Inter_inv = A_Inter_inv_set[fea_index, :, :]
    #
    #             # pull out the coor of the point
    #             b_target = np.array([i, j, 1])
    #
    #             # compute the barycentric coor
    #             barycentric = np.dot(A_Inter_inv, b_target)
    #
    #             # pull out the A_im matrix of the Tir index
    #             b_source = np.dot(A_im_set[fea_index, :, :], barycentric)
    #             assert len(b_source) == 3
    #             print(b_source[2])
    #
    #             b_source = b_source/b_source[2]
    #             b_source_map = np.round(b_source).astype(int)[0:2]
    #
    #             generated_pic[i, j, 0] = image[b_source_map[0], b_source_map[1], 0]
    #             generated_pic[i, j, 1] = image[b_source_map[0], b_source_map[1], 1]
    #             generated_pic[i, j, 2] = image[b_source_map[0], b_source_map[1], 2]

    return generated_pic


def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
    # TODO: Your code here
    # Tips: use Delaunay() function to get Delaunay triangulation;
    # Tips: use tri.find_simplex(pts) to find the triangulation index that pts locates in.

    # NOTICE: HERE WE ASSUME TWO IMAGE IS OF SAME SIZE!!
    # compute the H,W of the images
    size_H = im1.shape[0]
    size_W = im1.shape[1]

    morphed_im = np.zeros((len(warp_frac), size_H, size_W, 3))
    # for each film, we generate the warp image
    for i in range(len(warp_frac)):
        # find the coordinate of the intermediate point
        img_coor_inter = (1 - warp_frac[i]) * im1_pts + \
            warp_frac[i] * im2_pts
        # For each film create a new triangulation
        Tri = Delaunay(img_coor_inter)
        nTri = Tri.simplices.shape[0]  # #ofTri
        # print(Tri.points)
        # print(Tri.simplices)
        # print(Tri.find_simplex([299, 1]))
        # save all A matrix, A.shape = (3,3,nTri)
        A_Inter_inv_set = np.zeros((nTri, 3, 3))
        # print(A_Inter_inv_set)
        A_im1_set = np.zeros((nTri, 3, 3))
        A_im2_set = np.zeros((nTri, 3, 3))

        for ii, element in enumerate(Tri.simplices):
            # print(ii)
            # print(element)
            A_Inter_inv_set[ii, :, :] = np.linalg.inv(matrixA_for_three_points(img_coor_inter, element))
            A_im1_set[ii, :, :] = matrixA_for_three_points(im1_pts, element)
            A_im2_set[ii, :, :] = matrixA_for_three_points(im2_pts, element)
        assert A_Inter_inv_set.shape[0] == nTri
        # print(A_Inter_inv_set)
        # print(A_Inter_inv_set[0,:,:])

        # generate warp pictures
        warp_im1 = generate_warp(size_H, size_W,
                                 Tri, A_Inter_inv_set,
                                 A_im1_set, im1)
        warp_im2 = generate_warp(size_H, size_W,
                                 Tri, A_Inter_inv_set,
                                 A_im2_set, im2)
        # dissolve process
        dissolved_pic = (1 - dissolve_frac[i]) * warp_im1 + dissolve_frac[i] * warp_im2
        dissolved_pic = dissolved_pic.astype(np.uint8)
        # film.append(dissolved_pic)
        # save images
        # Im = Image.fromarray(dissolved_pic, 'RGB')
        # Im.save('./images/'+str(i)+'.png')
        morphed_im[i, :, :, :] = dissolved_pic.astype(np.uint8)
        # morphed_im
    return morphed_im


if __name__ == '__main__':
    # identify the target and source file
    source_file_name = 'source.jpg'
    target_file_name = 'target.jpg'
    # read in the target & source file
    im1 = np.array(Image.open(source_file_name).convert('RGB'))
    im2 = np.array(Image.open(target_file_name).convert('RGB'))
    # resize im1 and im2 to the identical size
    img_size_H = 300
    img_size_W = 300
    im1 = scipy.misc.imresize(im1, [img_size_H, img_size_W])
    im2 = scipy.misc.imresize(im2, [img_size_H, img_size_W])

    # generate coordinates
    im1_pts, im2_pts = click_correspondences(im1, im2)
    # print(im1_pts)
    # print(im2_pts)

    if im1_pts.shape != im2_pts.shape:
        print('the points number in two imgs are different')
    # define M films
    M = 60
    # generate warp_frac & dissolve_frac
    warp_frac = np.linspace(0, 1, num=M)

    dissolve_frac = np.linspace(0, 1, num=M)
    # dissolve_frac = np.array([0.5])
    # generate films
    # films_set = morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac)

    morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac)
