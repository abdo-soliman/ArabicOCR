import cv2
import numpy as np
from argparse import ArgumentParser
from segmentation import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--image', default="./DataSets/test.png", type=str, help='Image File Path')
    parser.add_argument(
        '-t', '--text', default="./DataSets/test.txt", type=str, help='Text File Path')

    args = parser.parse_args()
    img = cv2.imread(args.image)
    lines = breakLines(img)
