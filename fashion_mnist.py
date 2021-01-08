from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm.auto

tqdm.tqdm = tqdm.auto.tqdm


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['Футболка / топ', "Шорты", "Свитер", "Платье",
              "Плащ", "Сандали", "Рубашка", "Кроссовок", "Сумка",
              "Ботинок"]

train_images = train_images / 255.0
test_images = test_images / 255



# i = 0
# for (image, label) in test_dataset.take(25):
#     image = image.numpy().reshape((28, 28))
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image, cmap=plt.cm.binary)
#     plt.colorbar()
#     plt.grid(False)
#     plt.xlabel(class_names[label])
#     i += 1
#
#
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax),
])

model.compile(loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')

model.fit(train_images, train_labels, epochs=5)

predictions = model.predict(test_images)

win = np.argmax(predictions[1])

plt.figure()
plt.imshow(test_images[1], cmap=plt.cm.binary)
plt.grid(False)
plt.xlabel(class_names[win])
plt.show()




















