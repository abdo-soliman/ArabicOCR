from skimage import io
from skimage import filters, color
from skimage import transform
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import numpy as np
import os
import fnmatch


def dir2filename(dirName):
    # read files from directory
    matches = []
    for root, dirnames, filenames in os.walk(dirName):
        for filename in fnmatch.filter(filenames, '*.[jp]*'):
            matches.append(os.path.join(root, filename))
            print('({}/{})'.format(root, filename))

    print("Number of image files", len(matches))

    return matches


def img2fv(fileName, img_row, img_col):

    digit_y = fileName.split('-')[-1].split('.')[0]

    digit = io.imread(fileName)
    gray_image = color.rgb2gray(digit)
    thresh = filters.threshold_mean(gray_image)
    binary_image = gray_image > thresh

    # find an object from image
    label_objects, nb_labels = ndi.label(np.invert(binary_image))

    # create feature vector
    digit_x = np.zeros((1, (img_row*img_col)), dtype='float64')
    digit_fv = np.zeros((1, (img_row*img_col)+1), dtype='float64')

    for i in(range(1, nb_labels)):
        tmp = label_objects == i
        r, = np.where(tmp.sum(axis=1) > 1)
        c, = np.where(tmp.sum(axis=0) > 1)

        try:
            # crop with border
            tmp_img = gray_image[(r.min()-1):(r.max()+2), (c.min()-1):(c.max()+2)]
            digit_x = transform.resize(tmp_img, (img_row, img_col), mode='reflect')
            digit_x = digit_x.reshape((1, (img_row*img_col)))

            tmp_digit_data = np.hstack((digit_y, digit_x[0, :]))
            digit_fv = np.vstack((digit_fv, tmp_digit_data))
        except ValueError:
            pass

    # remove first row
    digit_fv = np.delete(digit_fv, 0, 0)

    return digit_fv

if __name__ == "__main__":
    # define size of the digit
    img_row = 25
    img_col = 25

    output_filename = 'digit_fv.train'
        
    # input directory
    dirName = 'RefactoedDataSet'
    matches = dir2filename(dirName)
    feature = np.zeros((1,(img_row*img_col)+1), dtype='float64')

    for i in(range(0,len(matches))):
        print("#", i+1)
        # create feature vector
        digit_fv = img2fv(matches[i], img_row, img_col)      
        feature = np.vstack((feature, digit_fv))

    # delete the first zero value row
    feature = np.delete(feature, 0, 0)
    print("Feature vector size", feature.shape)
        
    # save feature vector to text file
    print("Save feature vector to file", output_filename)
    with open(output_filename,"ab") as f:
        f.write("\n".join(" ".join(map(str, x)) for x in (feature)))
    f.close()
