from tensorflow import keras
from tensorflow.keras import callbacks
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Seed
SEED = 3
np.random.seed(SEED)

# Loading and transforming data to the right shape for the model
hm_X = pickle.load(open('../../data/handmade_datasets/X_handmade.pickle', 'rb'))
hm_y = pickle.load(open('../../data/handmade_datasets/y_handmade.pickle', 'rb'))

hm_X = hm_X.reshape((-1, 28, 28, 1))
hm_X_train, hm_X_test, hm_y_train, hm_y_test = train_test_split(hm_X, hm_y, test_size=0.2, shuffle=True)

(X_train, y_train), _ = mnist.load_data()
X_train = X_train.reshape((-1, 28, 28, 1))

# Adding random images of non-digits to train model to recognize whether it sees a digit or not
X_random = np.random.randint(0, 256, size=(1000, 28, 28, 1))
y_random = np.ones((10000,)) * 10

(X_fashion, _), _ = fashion_mnist.load_data()
X_fashion = X_fashion.reshape((-1, 28, 28, 1))

line = np.linspace(0, 128, num=784)

sin_random = []
sin = np.sin(line).reshape((28, 28))

for _ in range(1000):
    sin_random.append(sin + np.random.randn(28, 28))

sin_random = np.array(sin_random).reshape((-1, 28, 28, 1))
sin_random = ((sin_random - sin_random.min()) / (sin_random.max() - sin_random.min()))*255


# Shuffling MNIST data
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

# Concatenating all the data together
X_train = np.array(list(X_train)[:3000] + list(hm_X_train) + list(X_random)[:200] +
                   list(X_fashion)[:500] + list(sin_random)[:300])
y_train = np.array(list(y_train)[:3000] + list(hm_y_train) + list(y_random)[:200] +
                   list(y_random)[:500] + list(y_random)[:300])

X_test = np.array(list(hm_X_test)+list(X_fashion)[150:155])
y_test = np.array(list(hm_y_test)+list(y_random)[150:155])

X_train, X_test = X_train / 255, X_test / 255
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator()


# Building the model
def build_model():
    model = keras.Sequential()

    # Convolutional part
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Dense part
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(11))

    model.compile(optimizer='adam', metrics=['acc'], loss=keras.losses.CategoricalCrossentropy(from_logits=True))

    return model


def get_callbacks():
    fit_callbacks = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            cooldown=10,
            min_lr=1e-5,
            verbose=1,
        ),

        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
        ),
    ]

    return fit_callbacks


model = build_model()
fit_callbacks = get_callbacks()

X_test, X_val, y_test, y_val, = train_test_split(X_test, y_test, test_size=0.2, shuffle=True)

BATCH = 256
VAL_BATCH = 8
train_flow = train_datagen.flow(X_train, y_train, batch_size=BATCH, seed=SEED, shuffle=True)
val_flow = val_datagen.flow(X_val, y_val, batch_size=VAL_BATCH, seed=SEED, shuffle=True)

history = model.fit(train_flow, epochs=45, validation_data=val_flow, callbacks=fit_callbacks,
                    steps_per_epoch=len(X_train) // BATCH, validation_steps=len(X_val) // VAL_BATCH)

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
if acc > 0.995:
    path = 'saved_models/best_model.h5'
    model.save(path)
