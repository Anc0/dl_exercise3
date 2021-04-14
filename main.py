import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

from data import Processor
from network import Network

IMAGE_SIZE = (256, 256)

p = Processor(train_path="toy_train", test_path="toy_test1")
X_train, y_train, X_test, y_test = p.import_data()

n = Network()
model = n.generate_model()

y_train_hot = p.one_hot_class(y_train)

model.compile(optimizer="adam", loss="categorical_crossentropy")
model.fit(X_train, y_train_hot, epochs=10)
