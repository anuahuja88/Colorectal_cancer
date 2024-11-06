import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras import layers, models


print(tf.__version__)
data_set = "Data"

gpus = tf.config.experimental.list_physical_devices('GPU')
print(len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data = tf.keras.utils.image_dataset_from_directory(data_set)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
# Save the first image to disk
cv2.imwrite("test_image.png", batch[0][0])
test = cv2.imread("test_image.png")
converted = cv2.cvtColor(test, cv2.COLOR_RGB2BGR)
cv2.imshow('TEST', converted)
cv2.waitKey(0)
cv2.destroyAllWindows()


scaled_data = data.map(lambda x, y: (x /255, y))
scaled_batch = scaled_data.as_numpy_iterator().next()

print(len(data))
train_size = int(len(scaled_data) * 0.7)
val_size = int(len(scaled_data) * 0.2)
test_size = int(len(scaled_data) * 0.1) + 1


train = scaled_data.take(train_size)
val = scaled_data.skip(train_size).take(val_size)
test = scaled_data.skip(train_size + val_size).take(test_size)



print("Batch shape:", scaled_batch[0].shape) 
print("Labels:", scaled_batch[1])  


model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# model = keras.Sequential()
# model.add(layers.Conv2D(16, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(16, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

print(model.summary())

logdir = 'logs'
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])