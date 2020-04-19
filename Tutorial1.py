from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


data = tf.keras.datasets.mnist
(train_images, train_labes), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labes, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Acc: ", test_acc)

prediction = model.predict(test_images)
for i in range(5):
	plt.grid(False)
	plt.imshow(test_images)
	plt.xlabel("Actual: " + class_names[test_labels[i]])
	plt.title("Prediction" + class_names)
	plt.show()