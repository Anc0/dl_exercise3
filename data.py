from os import listdir

import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder


class Processor:

    def __init__(self, path="data/", train_path="train/", test_path="test1/", image_size=(256, 256)):
        self.path = path
        self.train_path = path + train_path
        self.test_path = path + test_path
        self.image_size = image_size

    def import_data(self):
        """
        Import testing and training data of the given dataset.
        """
        X_train, y_train = self.iterate_dir(self.train_path)
        X_test, y_test = self.iterate_dir(self.test_path)

        return X_train, y_train, X_test, y_test

    def iterate_dir(self, directory):
        """
        Iterate through a given directory and return images and their classes.
        """
        dirs = listdir(directory)
        X = []
        y = []
        labels = []
        i = 0
        # Open every image and save it in a list, including the class labels
        for dir in dirs:
            for impath in listdir(directory + "/" + dir):
                image = io.imread(directory + "/" + dir + "/" + impath)
                image = self.resize_image(image)

                X.append(image)
                labels.append(dir)
                y.append(i)
            i += 1
        X = np.array(X)
        y = np.array(y)
        X = X / 255
        return X, y

    def resize_image(self, image):
        """
        Resize an image to a given size.
        """
        return resize(image, self.image_size, anti_aliasing=True)

    @staticmethod
    def one_hot_class(y):
        """
        One hot encode the class labels (which are 0-start incremented integers).
        """
        enc = OneHotEncoder(sparse=False)
        y = y.reshape(-1, 1)
        return enc.fit_transform(y)
