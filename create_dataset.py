import time
from getkeys import key_check, keys_to_output
from grab_screen import process_image
import numpy as np
from random import shuffle


filename = 'training_data.npy'
# if os.path.isfile(filename):
#     training_data = list(np.load(filename))
#     print('Creating New file..')
# else:
#     training_data = []
#     print('Loading previous one..')

forward = []
left = []
right = []
forward_left = []
forward_right = []


def create_dataset():
    for i in range(5, 0, -1):
        print('In', i)
        time.sleep(1)

    while True:
        data_vector = process_image().flatten()
        output = keys_to_output(key_check())

        #           [A, W, D, AW, DW]
        if output == [1, 0, 0, 0, 0] and len(left) < 5000:
            left.append([data_vector, output])
        elif output == [0, 1, 0, 0, 0] and len(forward) < 5000:
            forward.append([data_vector, output])
        elif output == [0, 0, 1, 0, 0] and len(right) < 5000:
            right.append([data_vector, output])
        elif output == [0, 0, 0, 1, 0] and len(forward_left) < 5000:
            forward_left.append([data_vector, output])
        elif output == [0, 0, 0, 0, 1] and len(forward_right) < 5000:
            forward_right.append([data_vector, output])
        else:
            pass

        print('Left:', len(left), 'Forward:', len(forward), 'Right:', len(right),
              'Forward_left', len(forward_left), 'Forward_right:', len(forward_right))
        if len(left) == len(forward) == len(right) == len(forward_left) == len(forward_right) == 5000:
            shuffle(left)
            shuffle(forward)
            shuffle(right)
            shuffle(forward_left)
            shuffle(forward_left)

            print('Saving..')
            np.save('left.npy', left)
            np.save('forward.npy', forward)
            np.save('right.npy', right)
            np.save('forward_left.npy', forward_left)
            np.save('forward_right.npy', forward_right)

            training_data = left + forward + right + forward_left + forward_right
            shuffle(training_data)
            np.save(filename, training_data)
            break


create_dataset()
