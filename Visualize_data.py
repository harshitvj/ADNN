import cv2
import numpy as np
from time import sleep

filename = 'right.npy'
train_data = np.load(filename)

for i in range(len(train_data)):
    print(train_data[i][0])
    cv2.imshow('Current Frame', train_data[i][0].reshape(23, 96))
    print(train_data[i][1])
    sleep(0.3)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
