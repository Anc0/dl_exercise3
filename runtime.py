import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.python.keras.models import load_model
import plotly.express as px


class Trainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self, model, params):
        """
        Train the given model.
        """
        model.compile(optimizer=params["optimiser"], loss="categorical_crossentropy", metrics="accuracy")
        model.fit(self.X, self.y, epochs=params["epochs"], batch_size=params["batch_size"])
        return model


class Predictor:
    def __init__(self, model):
        self.model = load_model(model)

    def show_results(self, X, y):
        """
        Display some basic results.
        """
        y_pred = np.argmax(self.predict(X), axis=1)
        print(y_pred)
        print(accuracy_score(y, y_pred))

        fig = px.imshow(confusion_matrix(y, y_pred))
        fig.show()

    def predict(self, X):
        """
        Predict the signs in one of possible categories. Also implement a discriminator
        that checks if a given sign even belongs into any of the categories.
        """
        return self.model.predict(X)
