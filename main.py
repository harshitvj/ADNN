# from PIL import ImageGrab
import cv2
import numpy as np
from grab_screen import grab_screen
import time
from roi import roi
from getkeys import key_check, keys_to_output
import os
import  pandas as pd


lastTime = time.time()
frame = 0

# filename = 'training_data.npy'
#
# if os.path.isfile(filename):
#     print("Loading previous file..")
#     training_data = np.array(np.load(filename))
# else:
#     print("File doesn't exist, creating new file")
#     training_data = np.array([])

# Training Data Variable
data = pd.DataFrame()

while True:
    if time.time() - lastTime > 1:
        print('Frames per Second : '+str(frame))
        lastTime = time.time()
        frame = 0
    # grabbed_image = np.array(ImageGrab.grab(bbox=(0, 250, 960, 450)))
    grabbed_image = grab_screen(region=(0, 250, 960, 480))
    gray_image = cv2.cvtColor(grabbed_image, cv2.COLOR_BGR2GRAY)
    cropped_image = np.array(roi(gray_image))
    # screen = process_image(grabbed_image)
    # lines = cv2.HoughLinesP(screen, 1, np.pi/180, 100, None, minLineLength=150, maxLineGap=0)
    # screen = show_lines(cv2.cvtColor(grabbed_image,cv2.COLOR_BGR2RGB), lines)
    cropped_image = cv2.resize(cropped_image, (96, 23))
    data_vector = cropped_image.flatten()
    output = keys_to_output(key_check())
    # with open('yoklo.txt', 'a+') as file1:
    #     file1.write(str(data_vector))
    #     file1.write(" ")
    #     file1.write(str(output))
    #     file1.write("\n")

    # Pandas DataFrames for x_data and y_data for Training Data
    dfx = pd.DataFrame(data_vector)
    dfy = pd.DataFrame(output)
    # dfz = pd.concat([dfx,dfy])
    # data = pd.concat([data,dfz], axis=1)
    data = pd.concat([data,pd.concat([dfx,dfy])], axis=1)

    cv2.imshow('Lanes', cropped_image)

    frame += 1
    if cv2.waitKey(50) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# Training Data to a CSV file. Each row of file is a training example of type (x1,x2,...,y1,y2,y3,y4)
data.T.to_csv('data.csv',mode='a',header=False)
