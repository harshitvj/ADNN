import cv2
import numpy as np
import pandas as pd
from time import sleep

# # Read data from CSV file to DataFrame variable and convert its data type to int32
# data = pd.read_csv('data.csv', index_col=0)
# data = data.astype(np.uint8)
# data = data.reset_index(drop=True)
# data = data.T.reset_index(drop=True).T
#
# x_data = data.iloc[:, :-9].values
# y_data = data.iloc[:, -9:].values
filename = 'training_data.npy'
train_data = np.load(filename)

for i in range(len(train_data)):
    print(train_data[i][0])
    cv2.imshow('Current Frame', train_data[i][0].reshape(23, 96))
    print(train_data[i][1])
    # sleep(0.5)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
