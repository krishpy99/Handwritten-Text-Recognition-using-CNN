import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates

def coordinates_remap(image, factor_alpha, factor_sigma):
    alpha = image.shape[1] * factor_alpha
    sigma = image.shape[1] * factor_sigma
    shape = image.shape
    blur_size = int(4*sigma) | 1
    dx = alpha * cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)
    dy = alpha * cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return np.array(map_coordinates(image, indices, order=1, mode='constant').reshape(shape)) 