import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from six.moves import urllib

# Download dataset
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
(train_img, train_label), (test_img, test_label) = mnist.load_data()

train_img = train_img.reshape(-1, 28, 28, 1) / 255
train_label = train_label.reshape(-1, 1)
test_img = test_img.reshape(-1, 28, 28, 1) / 255
test_label = test_label.reshape(-1, 1)
print('train_img: ' + str(train_img.shape))
print('train_label: ' + str(train_label.shape))
print('test_img:  '  + str(test_img.shape))
print('test_label:  '  + str(test_label.shape))

train_label = keras.utils.to_categorical(train_label, 10)
test_label = keras.utils.to_categorical(test_label, 10)

# Create model
inputs = keras.Input(shape=(28,28,1))
x = keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation="relu")(inputs)
x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2, 2))(x)
x = keras.layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation="relu")(x)
x = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1024, activation="relu")(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs, name="MNIST_classifier")
model.summary()

# Train
model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

batch_size = 64
epochs = 10
history = model.fit(train_img, train_label,
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_data=(test_img, test_label)
                    )

# Evaluation
score = model.evaluate(test_img, test_label, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save model
model.save('float_model.h5')

