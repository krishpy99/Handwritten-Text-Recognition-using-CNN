import matplotlib.pyplot as plt
import numpy as np
import cv2

SMALL_HEIGHT = 800

def implt(img, cmp=None, t=''):
    plt.imshow(img, cmap=cmp)
    plt.title(t)
    plt.show()

def resize(img, height=SMALL_HEIGHT, always=False):
    if (img.shape[0] > height or always):
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))
    return img

def ratio(img, height=SMALL_HEIGHT):
    return img.shape[0] / height

def img_extend(img, shape):
    x = np.zeros(shape, np.uint8)
    x[:img.shape[0], :img.shape[1]] = img
    return x