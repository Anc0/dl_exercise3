from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt


class Processor:

    def __init__(self, path="data/", train_path="train/", test_path="test1/"):
        self.path = path

        self.train_path = path + train_path
        self.test_path = path + test_path

    def import_data(self):
        """
        Import testing and training data of the given dataset.
        """
        BATCH_SIZE = 64
        train = image_dataset_from_directory(self.train_path, batch_size=BATCH_SIZE)
        test = image_dataset_from_directory(self.test_path, batch_size=BATCH_SIZE)
        return train, test

    def resize_image(self, image):
        """
        Resize an image to a given size. Usually that is the size of the smallest image in the datset.
        """
        pass



