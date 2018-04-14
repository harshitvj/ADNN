import time
import cv2
import pandas as pd
from getkeys import key_check, keys_to_output
from grab_screen import process_image
import os
import numpy as np


filename = 'training_data.npy'
if os.path.isfile(filename):
    training_data = list(np.load(filename))
    print('Creating New file..')
else:
    training_data = []
    print('Loading previous one..')


def create_dataset():
    for i in range(5, 0, -1):
        print('In', i)
        time.sleep(1)

    count = 0
    total = 0
    while True:
        data_vector = process_image().flatten()
        output = keys_to_output(key_check())

        training_data.append([data_vector, output])

        # Pandas DataFrames for x_data and y_data for Training Data
        # data = pd.DataFrame()
        # dfx = pd.DataFrame(data_vector)
        # dfy = pd.DataFrame(output)

        # dfz = pd.concat([dfx,dfy])
        # data = pd.concat([data,dfz], axis=1)
        # data = pd.concat([data, pd.concat([dfx, dfy])], axis=1)
        # data.T.to_csv('data_new.csv', mode='a', header=False)

        count += 1
        if count % 500 == 0:
            total += 500
            count = 0
            print('total frames grabbed : {}'.format(total))
            np.save(filename, training_data)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


create_dataset()
