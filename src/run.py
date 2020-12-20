import pygame as pg
import numpy as np
from modeling.prediction import pred_digit
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

pg.init()

pg.display.set_caption("Digit recognizer")

screen = pg.display.set_mode((840, 840))

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
screen.fill(BLACK)
pixel_size = 25

is_button_pressed = False
to_classify = False
is_running = True
while is_running:

    for event in pg.event.get():
        if event.type == pg.QUIT:
            is_running = False
        elif event.type == pg.KEYDOWN and event.key == pg.K_RETURN:
            to_classify = True
        elif event.type == pg.MOUSEBUTTONDOWN:
            is_button_pressed = True
            if event.button == pg.BUTTON_LEFT:
                pg.draw.rect(screen, WHITE, (list(np.array(event.pos) - pixel_size), (pixel_size * 2, pixel_size * 2)),
                             0)
        elif event.type == pg.MOUSEBUTTONUP:
            is_button_pressed = False

        elif event.type == pg.MOUSEMOTION and is_button_pressed:
            pg.draw.rect(screen, WHITE,
                         ((event.pos[0] - pixel_size, event.pos[1] - pixel_size), (pixel_size * 2, pixel_size * 2)), 0)

    pg.display.update()

    if to_classify:
        pg.image.save(screen, 'tmp/user_digit.png')

        pred_digit()

        screen.fill(BLACK)
        to_classify = False


pg.display.quit()
pg.quit()
print('\nBye!')
