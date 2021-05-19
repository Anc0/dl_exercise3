from os import listdir

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.python.keras.models import load_model
import plotly.express as px
from plotly import graph_objects as go


class Trainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.model_name = None

    def train(self, model, params, verbose=1):
        """
        Train the given model.
        """
        model.compile(optimizer=params["optimiser"], loss="categorical_crossentropy", metrics="accuracy")
        history = model.fit(self.X, self.y, epochs=params["epochs"], batch_size=params["batch_size"])
        if verbose:
            self.display_acc_loss(history, params["epochs"])

        self.save_model(model)
        return model

    def display_acc_loss(self, history, epochs):
        """
        Show a graph of loss and accuracy through epochs.
        """
        for param in ["accuracy", "loss"]:
            fig = go.Figure(data=go.Scatter(x=[x for x in range(epochs)], y=[y for y in history.history[param]]))
            fig.update_layout(
                title=param.capitalize(),
                xaxis_title="Epochs",
                yaxis_title=param.capitalize()
            )
            fig.show()

    def save_model(self, model):
        """
        Save the model and increment the number in the name not to delete previous models.
        """
        models = sorted(listdir("models/"), reverse=True)
        self.model_name = "model_{}".format(str(int(models[0][-2:]) + 1).zfill(2))
        model.save("models/" + self.model_name)
        print("Model saved as {}.".format(self.model_name))



class Predictor:
    def __init__(self, model):
        self.model = load_model(model)

    def show_results(self, X, y, threshold=0.9):
        """
        Display some basic results.
        """
        y_pred = self.predict(X, threshold)
        y = [x+1 for x in y]
        y_pred = [x+1 for x in y_pred]

        fig = px.imshow(confusion_matrix(y, y_pred))
        fig.show()
        print(accuracy_score(y, y_pred))
        return y, y_pred

    def predict(self, X, threshold):
        """
        Predict the signs in one of possible categories. Also implement a discriminator
        that checks if a given sign even belongs into any of the categories.
        """
        # Get raw per class predictions
        predictions = self.model.predict(X)
        # If predicted class have a lower probability than threshold value, set it to the unknown class.
        unknowns = []
        for i, p in enumerate(predictions):
            if np.max(p) <= threshold:
                unknowns.append(i)
        y_pred = np.argmax(predictions, axis=1)
        for i in unknowns:
            # y_pred[i] = len(predictions[0])
            y_pred[i] = 50

        return y_pred
