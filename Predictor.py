import time
from Controller import Controller
from grab_screen import process_image
import tensorflow as tf
import numpy as np
from getkeys import key_check
from directkeys import ReleaseKey, W, A, D

# Defining Parameters
total_epochs = 50
learning_rate = 0.06

input_layer_nodes = 2208
hidden1_layer_nodes = 500
hidden2_layer_nodes = 500
hidden3_layer_nodes = 500
output_layer_nodes = 5

model_path = "./Model"

# Declaring variables and placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, input_layer_nodes], name='X_Placeholder')
Y = tf.placeholder(dtype=tf.float32, shape=[None, output_layer_nodes], name='Y_Placeholder')

weights = {
    'h1': tf.Variable(tf.truncated_normal([input_layer_nodes, hidden1_layer_nodes], stddev=0.01),
                      dtype=tf.float32, name='theta1'),
    'h2': tf.Variable(tf.truncated_normal([hidden1_layer_nodes, hidden2_layer_nodes], stddev=0.01),
                      dtype=tf.float32, name='theta2'),
    'h3': tf.Variable(tf.truncated_normal([hidden2_layer_nodes, hidden3_layer_nodes], stddev=0.01),
                      dtype=tf.float32, name='theta3'),
    'out': tf.Variable(tf.truncated_normal([hidden3_layer_nodes, output_layer_nodes], stddev=0.01),
                       dtype=tf.float32, name='theta4')
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([hidden1_layer_nodes], stddev=0.01),
                      dtype=tf.float32, name='bias1'),
    'b2': tf.Variable(tf.truncated_normal([hidden2_layer_nodes], stddev=0.01),
                      dtype=tf.float32, name='bias2'),
    'b3': tf.Variable(tf.truncated_normal([hidden3_layer_nodes], stddev=0.01),
                      dtype=tf.float32, name='bias3'),
    'out': tf.Variable(tf.truncated_normal([output_layer_nodes], stddev=0.01),
                       dtype=tf.float32, name='bias4')
}


def multi_layer_perceptron(x, weight, bias):
    layer_h1 = tf.add(tf.matmul(x, weight['h1']), bias['b1'])
    layer_h1 = tf.nn.sigmoid(layer_h1)

    layer_h2 = tf.add(tf.matmul(layer_h1, weight['h2']), bias['b2'])
    layer_h2 = tf.nn.sigmoid(layer_h2)

    layer_h3 = tf.add(tf.matmul(layer_h2, weight['h3']), bias['b3'])
    layer_h3 = tf.nn.relu(layer_h3)

    layer_out = tf.add(tf.matmul(layer_h3, weight['out']), bias['out'])

    return layer_out


init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, model_path)

hypothesis = multi_layer_perceptron(X, weights, biases)

for i in range(5, 0, -1):
    print('Test Run in ', i)
    time.sleep(1)

paused = False
while True:
    if not paused:
        data_vector = process_image().flatten()
        data_vector = np.float32(data_vector)

        prediction = tf.argmax(hypothesis, axis=1)
        kb = Controller()
        prediction = sess.run(prediction, feed_dict={X: data_vector.reshape(1, 2208)})[0]
        print(prediction)
        kb.act(prediction)

    keys = key_check()
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
