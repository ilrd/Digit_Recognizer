import pygame as pg
import numpy as np
from PIL import Image
import random
import pickle
import os

pg.init()

saving_path = '../../data/handdrawn_digits/'

pg.display.set_caption(f"Digit to draw - {str(len(os.listdir(saving_path)))[-1]}")

screen = pg.display.set_mode((840, 840))


is_running = True
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

screen.fill(BLACK)
is_left_down = False
to_save = False
saving_iterator = len(os.listdir(saving_path))

pixel_size = 30  # can be from ~20 to ~30


while is_running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            is_running = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button == pg.BUTTON_LEFT:
                is_left_down = True
                pg.draw.rect(screen, WHITE, (list(np.array(event.pos) - pixel_size), (pixel_size * 2, pixel_size * 2)), 0)
        elif event.type == pg.MOUSEBUTTONUP:
            is_left_down = False
        elif event.type == pg.MOUSEMOTION:
            if is_left_down:
                pg.draw.rect(screen, WHITE, ((event.pos[0] - pixel_size, event.pos[1] - pixel_size), (pixel_size * 2, pixel_size * 2)), 0)
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                to_save = True
    if to_save:
        to_save = False
        digit = int(str(saving_iterator)[-1])
        next_digit = int(str(saving_iterator + 1)[-1])
        pg.image.save(screen, f'{saving_path}{saving_iterator}.png')
        print(f'Saved digit {digit} to {saving_iterator}.png')
        screen.fill(BLACK)
        pixel_size = random.choice(list(range(25, 31)))
        pg.display.set_caption(f"Draw digit {next_digit}")
        saving_iterator += 1

    pg.display.update()

pg.display.quit()
pg.quit()

# Saving handmade data to pickle objects
X = []
y = []

for i in range(saving_iterator):
    im = Image.open(f'{saving_path}{i}.png')
    im = im.resize((28, 28))
    im_array = np.array(im)
    im_array = np.array([[np.average(j) for j in im_array[i]] for i in range(len(im_array))])

    digit = int(str(i)[-1])
    X.append(im_array)
    y.append(digit)

X = np.array(X)
y = np.array(y)

with open('../../data/handmade_datasets/y_handmade.pickle', 'wb') as f:
    pickle.dump(y, f)

with open('../../data/handmade_datasets/X_handmade.pickle', 'wb') as f:
    pickle.dump(X, f)
