import time
import cv2
import pandas as pd
from getkeys import key_check, keys_to_output
from grab_screen import process_image


def create_dataset():
    for i in range(5, 0, -1):
        print(i)
        time.sleep(1)
    count = 0
    total = 0
    while True:
        data = pd.DataFrame()
        data_vector = process_image().flatten()
        output = keys_to_output(key_check())

        # Pandas DataFrames for x_data and y_data for Training Data
        dfx = pd.DataFrame(data_vector)
        dfy = pd.DataFrame(output)

        # dfz = pd.concat([dfx,dfy])
        # data = pd.concat([data,dfz], axis=1)
        data = pd.concat([data, pd.concat([dfx, dfy])], axis=1)
        data.T.to_csv('data.csv', mode='a', header=False)

        count += 1
        if count % 500 == 0:
            total += 500
            count = 0
            print('total frames grabbed : {}'.format(total))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


create_dataset()
