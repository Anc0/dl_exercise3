from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class Network:

    def __init__(self, image_shape=(256, 256, 3), class_num=50):
        self.image_shape = image_shape
        self.class_num = class_num

    def generate_model(self):
        """
        Return a Keras model, ready for training.
        """

        in_layer = Input(shape=self.image_shape)
        x = Conv2D(32, kernel_size=4, activation="relu")(in_layer)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, kernel_size=4, activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, kernel_size=4, activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(32, activation="relu")(x)
        out_layer = Dense(self.class_num, activation="softmax")(x)

        return Model(inputs=in_layer, outputs=out_layer)
