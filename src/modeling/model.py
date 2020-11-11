from tensorflow import keras
import tensorflow as tf
from keras.datasets import mnist
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Loading and transforming data to the right shape for the model
hm_X = pickle.load(open('../../data/handmade_datasets/X_handmade.pickle', 'rb'))
hm_y = pickle.load(open('../../data/handmade_datasets/y_handmade.pickle', 'rb'))

hm_X = hm_X.reshape((-1, 28, 28, 1))
hm_X_train, hm_X_test, hm_y_train, hm_y_test = train_test_split(hm_X, hm_y, test_size=0.1, shuffle=True)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train.reshape((-1, 28, 28, 1)), X_test.reshape((-1, 28, 28, 1))

X_train = np.array(list(X_train)[:3000] + list(hm_X_train))
y_train = np.array(list(y_train)[:3000] + list(hm_y_train))
X_test = np.array(list(X_test)[:100] + list(hm_X_test))
y_test = np.array(list(y_test)[:100] + list(hm_y_test))

X_train, X_test = X_train / 255, X_test / 255
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Building the model
model = keras.Sequential()

# Convolutional part
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Dense part
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(10))


def schedule(epochs):
    if epochs >= 25:
        return 0.0001
    else:
        return 0.001


scheduler = LearningRateScheduler(schedule=schedule)

model.compile(optimizer='adam', metrics=['acc'], loss=keras.losses.CategoricalCrossentropy(from_logits=True))

model.summary()

X_val = X_test
y_val = y_test
history = model.fit(X_train, y_train, 32, 35, validation_data=(X_val, y_val), verbose=1, shuffle=True,
                    callbacks=[scheduler])

loss, acc = model.evaluate(X_test, y_test)

print(acc, '- accuracy on the testing data')


# ------------------------------------------#
# Plotting loss and acc during the training process

def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = np.array(history.history['acc'])
    val_acc = np.array(history.history['val_acc'])

    fig1 = plt.figure()
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    fig1.legend()
    fig1.show()

    fig2 = plt.figure()
    plt.plot(acc, label='acc')
    plt.plot(val_acc, label='val_acc')
    fig2.legend()
    fig2.show()


plot_history(history)

# ------------------------------------------#
# Saving the model
if acc > 0.9954:
    path = 'saved_models/best_model.h5'
    model.save(path)

