import glob
import cv2
import numpy as np
from utils import *
from argparse import ArgumentParser


Y_START = (cv2.imread("./templates/y_template.png", 0) / 255).astype(np.uint8)
SEN_START = (cv2.imread("./templates/sen_start_template.png",
                        0) / 255).astype(np.uint8)
H_START = (cv2.imread("./templates/7_template.png", 0) / 255).astype(np.uint8)
LETTER_COUNTS = {
    "ا": 0,
    "ب": 0,
    "ت": 0,
    "ث": 0,
    "ج": 0,
    "ح": 0,
    "خ": 0,
    "د": 0,
    "ذ": 0,
    "ر": 0,
    "ز": 0,
    "ش": 0,
    "س": 0,
    "ص": 0,
    "ض": 0,
    "ط": 0,
    "ظ": 0,
    "ع": 0,
    "غ": 0,
    "ف": 0,
    "ق": 0,
    "ك": 0,
    "ل": 0,
    "م": 0,
    "ن": 0,
    "ه": 0,
    "و": 0,
    "ي": 0,
    "ﻻ": 0,
    "ى": 0,
    "ئ": 0,
    "ء": 0,
    "ؤ": 0,
    "ة": 0
}


def breakLines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # (2) threshold
    th, threshed = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)

    # (3) minAreaRect on the nozeros
    H, W = img.shape[:2]
    (cx, cy), (w, h), ang = ret

    if (H > W and w > h) or (H < W and w < h):
        w, h = h, w
        ang += 90

    # ## (4) Find rotated matrix, do rotation
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))

    # (5) find and draw the upper and lower boundary of each lines
    hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)

    th = 2
    uppers = [y for y in range(H-1) if hist[y] <= th and hist[y+1] > th]
    lowers = [y for y in range(H-1) if hist[y] > th and hist[y+1] <= th]

    uppers_mod = [uppers[0]]
    lowers_mod = [lowers[0]]

    for i in range(1, len(uppers)):
        if (lowers[i] - lowers[i-1]) > 10:
            uppers_mod.append(uppers[i])
            lowers_mod.append(lowers[i])

    lines = []
    if len(uppers_mod) == len(lowers_mod):
        for i in range(len(uppers_mod)):
            lines.append(rotated[uppers_mod[i]-5:lowers_mod[i]+5, :])

    return lines


def breakWords(line):
    H, W = line.shape
    hist = cv2.reduce(line, 0, cv2.REDUCE_AVG).reshape(-1)

    lefts = []
    rights = []

    start = False
    count = 0
    possible_end = 0
    for i in range(len(hist)):
        if hist[i] != 0 and start:
            count = 0

        if hist[i] == 0 and start:
            if count == 0:
                possible_end = i
            if count >= 3:
                count = 0
                start = False
                rights.append(possible_end)
            count = count + 1

        if hist[i] != 0 and not start:
            start = True
            lefts.append(i)

    words = []
    if len(lefts) == len(rights):
        for i in range(len(rights)):
            words.append(line[:, lefts[i]:rights[i]])

    return words


def extractTemplate(template, skeleton=True):
    if not skeleton:
        template = toSkeleton(template)
    template = template[~np.all(template == 0, axis=1)]
    mask = (template == 0).all(0)
    template = template[:, ~mask]

    return template


def templateMatch(skeleton, template, threshold=0.4, start=False):
    if template.shape[0] > skeleton.shape[0] or template.shape[1] > skeleton.shape[1]:
        return 0

    w, h = template.shape[::-1]
    res = cv2.matchTemplate(skeleton.astype(np.uint8),
                            template.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        if start:
            return pt[0]
        return pt[0] + w

    return 0


def getBaseLine(skeleton):
    hist = getHist(skeleton, axes=1)
    base_line = np.where(hist == np.max(hist))[0][0]
    return base_line


def addEmptySpaceSep(hist):
    splitters = []
    not_empty = False
    for i in range(len(hist)):
        if not_empty and hist[i] == 0:
            splitters.append(i)
            not_empty = False

        if hist[i] != 0:
            not_empty = True

    return np.array(splitters)


def isFakeAlef(word, splitter):
    start = 0
    for i in range(word.shape[1]):
        if np.sum(word[:, i]) > 0:
            start = i
            break
    sub_word = word[:, start:splitter]

    if sub_word.shape[1] > 3:
        return False
    for i in range(sub_word.shape[1]):
        column = sub_word[:, i]
        indecies = np.where(column > 0)[0]

        if len(indecies) > 1 and (abs(indecies[-1] - indecies[0]) > 3):
            return False

    return True


def matchFirstCharacter(word):
    skeleton = toSkeleton(word)
    first_sen_sep = templateMatch(skeleton, SEN_START, threshold=0.55)
    first_y_sep = templateMatch(skeleton, Y_START, threshold=0.8)
    first_h_sep = templateMatch(skeleton, H_START, threshold=0.5)

    first_sep = 0
    if first_sen_sep != 0:
        first_sep = first_sen_sep
    elif first_y_sep != 0:
        first_sep = first_y_sep
    elif first_h_sep != 0:
        first_sep = first_h_sep

    return first_sep


def segmenteCharacters(word, to_skeleton=True, debug=False):
    full_skeleton = toSkeleton(word) if to_skeleton else word
    skeleton = toSkeleton(word) if to_skeleton else word
    base_line = getBaseLine(skeleton)
    splitters = np.array([]).astype(np.uint8)

    first_sep = matchFirstCharacter(word)
    if first_sep >= word.shape[1]:
        return [full_skeleton]

    if first_sep != 0:
        splitters = np.append(splitters, first_sep)
        word = word[:, first_sep:]
        skeleton = toSkeleton(word)

    full_hist = getHist(skeleton)
    hist = getHist(skeleton[0:base_line-1, :])
    diff = np.diff(hist)

    # add splitters at the start of region that doesn't contain pixels
    empty_space_splitters = addEmptySpaceSep(full_hist) + first_sep
    splitters = np.append(np.array(splitters), empty_space_splitters)

    # add splitter at point if you were above base line and return back to baseline
    base_line_splitters = [(i + first_sep) for i in range(1, len(diff))
                           if (diff[i-1] != 0 and diff[i] == 0 and hist[i] == 0)]
    splitters = np.append(splitters, base_line_splitters).astype(np.uint8)
    splitters.sort()    # sort splitters in ascending order

    if len(splitters) == 0:
        return [full_skeleton]

    start = 2 if len(
        splitters) > 1 and splitters[1] - splitters[0] <= 3 and splitters[1] in empty_space_splitters else 1
    mod_splitters = np.array(
        [splitters[1] if start == 2 else splitters[0]]).astype(np.uint8)
    real_splitters = [splitters[i] for i in range(
        start, len(splitters)) if splitters[i] - splitters[i-1] > 3]
    mod_splitters = np.append(mod_splitters, real_splitters).astype(np.uint8)

    if len(mod_splitters) == 0:
        return [full_skeleton]

    # remove separators with no characters between them
    non_character_filtered = []
    for i in range(len(mod_splitters)-1):
        sub_hist = full_hist[mod_splitters[i] -
                             first_sep:mod_splitters[i+1]-first_sep]
        if np.sum((sub_hist > 1) + 0) > 0:
            non_character_filtered.append(mod_splitters[i])
    if np.sum(full_hist[mod_splitters[-1]-first_sep:]) > 0:
        non_character_filtered.append(mod_splitters[-1])

    if len(non_character_filtered) == 0:
        return [full_skeleton]

    # remove fake alef at the beginning
    start = 1 if isFakeAlef(skeleton, non_character_filtered[0]) else 0
    fake_alef_splitter_filtered = np.array(
        non_character_filtered[start:]).astype(np.uint8)

    if len(fake_alef_splitter_filtered) == 0:
        return [full_skeleton]

    if debug:
        bgr_skeleton = cv2.cvtColor(
            (255 * full_skeleton).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for splitter in fake_alef_splitter_filtered:
            cv2.line(bgr_skeleton, (splitter, 0), (splitter,
                                                   bgr_skeleton.shape[1]), (255, 0, 0), 1)

        return bgr_skeleton
    else:
        characters = []
        characters.append(full_skeleton[:, 0:fake_alef_splitter_filtered[0]])
        for i in range(len(fake_alef_splitter_filtered)-1):
            characters.append(
                full_skeleton[:, fake_alef_splitter_filtered[i]:fake_alef_splitter_filtered[i+1]])
        characters.append(full_skeleton[:, fake_alef_splitter_filtered[-1]:])

        return characters


def imageToWords(img):
    words = []
    lines = breakLines(img)
    for line in lines:
        line_words = breakWords(line)
        line_words.reverse()
        for word in line_words:
            words.append(word)

    return words


def imgToDataSet(img_path, text_path, destination_path):
    img = cv2.imread(img_path)
    img_words = imageToWords(img)
    with open(text_path) as f:
        text = f.readlines()[0]
        text_words = text.split()
        index = 0
        for word in text_words:
            if index >= len(img_words):
                break
            chars = segmenteCharacters(img_words[index])
            if len(chars) == len(word):
                chars.reverse()
                for i in range(len(word)):
                    letter = 255*extractTemplate(chars[i])
                    if letter.shape[0] < 25 and letter.shape[1] < 25:
                        mask = np.zeros((25, 25))

                        vertical_start = int((25 - letter.shape[0]) / 2)
                        vertical_end = vertical_start + letter.shape[0]
                        horizontal_start = int((25 - letter.shape[1]) / 2)
                        horizontal_end = horizontal_start + letter.shape[1]

                        mask[vertical_start:vertical_end,
                             horizontal_start:horizontal_end] = letter
                        cv2.imwrite(destination_path + "/" + word[i] + "/" + word[i] + "_" + str(
                            LETTER_COUNTS[word[i]]) + ".png", mask)
                        LETTER_COUNTS[word[i]] += 1
            index += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--img_dir', default="./DataSets/scanned", type=str, help='Image Folder Path')
    parser.add_argument(
        '-t', '--text_dir', default="./DataSets/text", type=str, help='Text Folder Path')
    parser.add_argument(
        '-d', '--destination', default="./dataSet", type=str, help='DataSet Detination Folder Path')

    args = parser.parse_args()
    destination = args.destination
    img_dir = args.img_dir
    text_dir = args.text_dir
    images = glob.glob(img_dir + "/*")
    texts = glob.glob(text_dir + "/*")

    text_files = []
    for text in texts:
        text_files.append(text.split('/')[-1].split('.')[0])

    image_files = []
    for image in images:
        filename = image.split('/')[-1].split('.')[0]
        if filename in text_files:
            image_files.append(image.split('/')[-1])

    for image in image_files:
        img_path = img_dir + "/" + image
        text_path = text_dir + "/" + image.split('.')[0] + ".txt"
        imgToDataSet(img_path, text_path, destination)
