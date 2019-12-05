import cv2
import numpy as np
from argparse import ArgumentParser

from utils import show_images
from thinning import zhangSuen
from skimage.filters import threshold_otsu

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--image', default="./DataSets/word.png", type=str, help='Image path')
    
    args = parser.parse_args()
    img = cv2.imread(args.image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray[img_gray < 129] = 0
    img_gray = np.invert(img_gray)
    
    Otsu_Threshold = threshold_otsu(img_gray)   
    thresholded = img_gray < Otsu_Threshold    # must set object region as 1, background region as 0 !

    thinnned = zhangSuen(thresholded)
    show_images([thresholded, thinnned], ["original", "thinned"])
    # cv2.imwrite("thinnned.png", thinnned)
