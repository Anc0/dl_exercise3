from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class Network:

    def __init__(self, image_shape=(256, 256, 3)):
        self.image_shape = image_shape
        pass

    def generate_model(self):
        """
        Return a Keras model, ready for training.
        """

        in_layer = Input(shape=self.image_shape)
        x = Conv2D(32, kernel_size=4, activation="relu")(in_layer)
        x = MaxPooling2D(pool_size=(4, 4))(x)
        x = Conv2D(32, kernel_size=4, activation="relu")(x)
        x = MaxPooling2D(pool_size=(4, 4))(x)
        x = Flatten()(x)
        x = Dense(8, activation="relu")(x)
        out_layer = Dense(5, activation="softmax")(x)

        return Model(inputs=in_layer, outputs=out_layer)


n = Network()
model = n.generate_model()

print(model.summary())
