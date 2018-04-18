import time
from alexnet import alexnet
from directkeys import ReleaseKey, W, A, D
from grab_screen import process_image
from getkeys import key_check
import numpy as np
from Controller import Controller

WIDTH = 96
HEIGHT = 23
LR = 1e-3

MODEL_PATH = './Model_alex'

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_PATH)


def main():
    for i in range(5, 0, -1):
        print(i)
        time.sleep(1)

    paused = False
    while True:

        if not paused:
            # print('loop took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()
            screen = process_image()
            prediction = model.predict([screen.reshape(96, 23, 1)])[0]

            output = np.argmax(prediction)
            if output == 0 and prediction[0] < 0.80:
                output = 3
            if output == 2 and prediction[2] < 0.80:
                output = 4
            kb = Controller()
            kb.act(output)

            if output == 0:
                print('Left')
            elif output == 1:
                print('Forward')
            elif output == 2:
                print('Right')
            elif output == 3:
                print('Forward_left')
            elif output == 4:
                print('Forward_right')
            else:
                print('wut?')

        keys = key_check()

        # p pauses game and can get annoying.
        if 'P' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)


main()
