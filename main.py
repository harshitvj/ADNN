# from PIL import ImageGrab
import cv2
import time

lastTime = time.time()
frame = 0

# filename = 'training_data.npy'
#
# if os.path.isfile(filename):
#     print("Loading previous file..")
#     training_data = np.array(np.load(filename))
# else:
#     print("File doesn't exist, creating new file")

# Training Data Variable

while True:
    if time.time() - lastTime > 1:
        print('Frames per Second : ' + str(frame))
        lastTime = time.time()
        frame = 0

    frame += 1
    if cv2.waitKey(50) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# Training Data to a CSV file. Each row of file is a training example of type (x1,x2,...,y1,y2,y3,y4)
