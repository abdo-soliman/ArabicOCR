from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from thinning import zhangSuen
from matplotlib.pyplot import bar
import matplotlib.pyplot as plt
import cv2
import numpy as np


def show_images(images, titles=None):
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def getHist(img, axes=0):
    hist = []
    if axes == 0:
        for i in range(img.shape[1]):
            hist.append(np.sum(img[:, i]))
    else:
        for i in range(img.shape[0]):
            hist.append(np.sum(img[i, :]))

    return np.array(hist)


def showHist(hist):
    plt.figure()
    bar(np.arange(len(hist)), hist, width=0.8, align='center')


def thin(img):
    img = np.invert(img)

    Otsu_Threshold = threshold_otsu(img)
    # must set object region as 1, background region as 0 !
    thresholded = img < Otsu_Threshold

    return zhangSuen(thresholded)


def toSkeleton(img):
    binary = img.copy()
    binary[binary <= 127] = 0
    binary[binary > 127] = 1
    return (skeletonize(binary) + 0)
