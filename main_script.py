import cv2
import os
from morph_tri import morph_tri
from click_correspondences import click_correspondences
import numpy as np
import scipy
from PIL import Image
import imageio
# from images2gif import writeGif
#
# import glob
# import moviepy.editor as mpy


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

    #define M films
    M = 60
    # generate warp_frac & dissolve_frac
    warp_frac = np.linspace(0, 1, num=M)

    dissolve_frac = np.linspace(0, 1, num=M)
    # dissolve_frac = np.array([0.5])
    # generate films
    films_set = morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac)

    ## AVI file
    # image_folder = './images'
    # video_name = './video.gif'
    # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # images.sort(key=lambda x: int(os.path.splitext(x)[0]))
    # images_path = ['./images/'+img for img in images]
    # frame = cv2.imread(os.path.join(image_folder, images[0]))
    # height, width, layers = frame.shape
    #
    # video = cv2.VideoWriter(video_name, 0, 30, (width, height))
    #
    # for image in images:
    #     video.write(cv2.imread(os.path.join(image_folder, image)))
    #
    # cv2.destroyAllWindows()
    # video.release()

    ## Moviepy
    # image_folder = './images'
    # video_name = './video.gif'
    # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # images.sort(key=lambda x: int(os.path.splitext(x)[0]))
    # images_path = ['./images/'+img for img in images]
    # fps = 60
    # # file_list = glob.glob('*.png')  # Get all the pngs in the current directory
    # # list.sort(file_list, key=lambda x: int(
    # #     x.split('_')[1].split('.png')[0]))  # Sort the images by #, this may need to be tweaked for your use case
    # clip = mpy.ImageSequenceClip(images_path, fps=fps)
    # clip.write_gif('{}.gif'.format(video_name), fps=fps)

    ## imageio.mimsave
    film_list = [films_set[i, :, :, :].astype(np.uint8) for i in range(films_set.shape[0])]
    exportname = "output.gif"
    imageio.mimsave(exportname, film_list, 'GIF', duration=0.1)

