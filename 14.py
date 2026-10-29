# 8.
# Implement and train a Convolutional Neural Network (CNN) for an imageclassification problem
# (e.g., handwritten digits, facial expressions, medical
# images, plant disease, or traffic signs) and evaluate model performance using
# appropriate metrics and validation techniques.

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5,validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

pred = model.predict(x_test).argmax(axis=1)
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
