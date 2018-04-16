from alexnet import alexnet
import numpy as np
from sklearn.model_selection import train_test_split

training_data = np.load('training_data.npy')
WIDTH = 96
HEIGHT = 23
LR = 1e-3
EPOCHS = 2

x_data = [x[0] for x in training_data]
y_data = [x[1] for x in training_data]

train_X, test_X, train_Y, test_Y = train_test_split(x_data, y_data, test_size=0.20, random_state=675)

train_X = np.array(train_X).reshape(-1, WIDTH, HEIGHT, 1)
test_X = np.array(test_X).reshape(-1, WIDTH, HEIGHT, 1)

MODEL_PATH = './Model_alex'

model = alexnet(WIDTH, HEIGHT, LR)

model.fit(train_X, train_Y, EPOCHS, validation_set=({'input': test_X}, {'targets': test_Y}),
          show_metric=True, snapshot_step=100, run_id=MODEL_PATH)
model.save()