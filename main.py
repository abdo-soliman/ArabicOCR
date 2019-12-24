import codecs
import time
import cv2
import numpy as np
from argparse import ArgumentParser
from segmentation import *
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("train.csv")
X = df.drop(['label'], axis=1)
y = df['label']
# create a model
clf_tree = DecisionTreeClassifier(max_depth=277)
clf_tree.fit(X,y)

# # save the classifier
neigh = KNeighborsClassifier(n_neighbors=29)  
neigh.fit(X, y)

clf_svm = svm.SVC(kernel='linear', C = 1.0)
clf_svm.fit(X,y)

def imgToText(img_path, text_file_path, classifier="svm"):
    img = cv2.imread(img_path)
    chars_arr = imageToChars(img)
    words = []
    for chars in chars_arr:
        word = ""
        for char in chars:
            feature_vector = charToFeatureVector(char)
            if classifier == "svm":
                word += clf_svm.predict([feature_vector])
            elif classifier == "knn":
                word += neigh.predict([feature_vector])
            else:
                word += clf_tree.predict([feature_vector])

        words.append(word[0])

    f = codecs.open(text_file_path, "w", "utf-8")
    f.write(" ".join(words))
    f.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--img_dir', default="./test", type=str, help='Image Folder Path')
    parser.add_argument(
        '-t', '--text_dir', default="./output/text", type=str, help='Text Folder Path')
    parser.add_argument(
        '-c', '--classifier', default="svm", type=str, help='tree or svm or knn')

    args = parser.parse_args()
    img_dir = args.img_dir
    text_dir = args.text_dir
    classifier = args.classifier

    images = glob.glob(img_dir + "/*")
    times = []
    for img_path in images:
        image = img_path.split('/')[-1]
        text_path = text_dir + "/" + image.split('.')[0] + ".txt" 
        start_time = time.time()
        imgToText(img_path, text_path, classifier)
        times.append(str(time.time() - start_time)+"sec")
    
    f = open("./output/running_time.txt", "w")
    f.write("\n".join(times))
    f.close()
