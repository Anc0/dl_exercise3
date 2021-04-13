from matplotlib import pyplot as plt
from tensorflow import keras

from data import Processor
from network import Network

IMAGE_SIZE = (256, 256)

p = Processor()
n = Network()

train, test = p.import_data()

plt.figure(figsize=(10, 10))
for images, labels in train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
    plt.show()

model = n.make_model(input_shape=IMAGE_SIZE + (3,), num_classes=2)

epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train, epochs=epochs, callbacks=callbacks, # validation_data=val_ds,
)
