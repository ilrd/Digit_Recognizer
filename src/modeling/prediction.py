from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


path = './modeling/saved_models/best_model.h5'
if not os.path.exists(path):
    path = './src/modeling/saved_models/best_model.h5'  # for docker implementation

model = load_model(path)
# Mock prediction to turn on tensorflow on the startup
pred = model.predict(np.zeros((1, 28, 28, 1)))


def pred_digit():
    img_path = './tmp/user_digit.png'
    if not os.path.exists(img_path):
        img_path = './src/tmp/user_digit.png'  # for docker implementation
    im = Image.open(img_path)
    im = im.resize((28, 28))
    im_array = np.array(im)
    im_array = np.array([[np.average(j) for j in im_array[i]] for i in range(len(im_array))])
    im_array = im_array / 255

    im_array = im_array.reshape((28, 28, 1))

    pred = model.predict(np.array([im_array]))
    pred_num = np.argmax(pred)

    if pred_num != 10:
        print(f"The digit you entered is {pred_num}")
    else:
        print(f"Not a digit")

