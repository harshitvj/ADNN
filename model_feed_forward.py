import tensorflow as tf
import os
import pandas as pd
# import numpy as np
from grab_screen import process_image

epochs = 100
learning_rate = 0.01

input_layer_nodes = 2208
hidden1_layer_nodes = 500
hidden2_layer_nodes = 500
output_layer_nodes = 9

# Read data from CSV file to DataFrame variable and convert its dtype to int32
data = pd.read_csv('data.csv', index_col=0)
data = data.astype('int32')
data = data.reset_index(drop=True)
data = data.T.reset_index(drop=True).T

x_data = data.iloc[:, :-output_layer_nodes].values
y_data = data.iloc[:, -output_layer_nodes:].values


def train_data():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Convert DataFrame "data" to numpy matrix and store x_data and y_data separately

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    theta1 = tf.Variable(tf.random_uniform([input_layer_nodes, hidden1_layer_nodes],
                                           minval=-0.01,  maxval=0.01, name='theta1'))
    theta2 = tf.Variable(tf.random_uniform([hidden1_layer_nodes, hidden2_layer_nodes],
                                           minval=-0.01, maxval=0.01, name='theta2'))
    theta3 = tf.Variable(tf.random_uniform([hidden2_layer_nodes, output_layer_nodes],
                                           minval=-0.01, maxval=0.01, name='theta3'))

    bias1 = tf.Variable(tf.zeros([hidden1_layer_nodes]), name='bias1')
    bias2 = tf.Variable(tf.zeros([hidden2_layer_nodes]), name='bias2')
    bias3 = tf.Variable(tf.zeros([output_layer_nodes]), name='bias3')

    layer2 = tf.sigmoid(tf.matmul(X, theta1) + bias1)
    layer3 = tf.sigmoid(tf.matmul(layer2, theta2) + bias2)
    hypothesis = tf.sigmoid(tf.matmul(layer3, theta3) + bias3)

    term1 = -(Y * tf.log(hypothesis))
    term2 = -((1-Y) * tf.log(1-hypothesis))
    cost = tf.reduce_mean(term1 + term2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init_var = tf.global_variables_initializer()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./logs/FeedForward', sess.graph)
        sess.run(init_var)
        for i in range(epochs):
            sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
            if i % 1 == 0:
                print(sess.run(cost, feed_dict={X: x_data, Y: y_data}))
                temp = sess.run([hypothesis, Y], feed_dict={X: x_data, Y: y_data})
                answer = tf.equal(tf.floor(temp[0] + 0.5), temp[1])
                accuracy = tf.reduce_mean(tf.cast(answer, tf.float32))
                print(sess.run(accuracy, feed_dict={X: x_data, Y: y_data})*100)

    parameters = {'theta1': theta1, 'theta2': theta2, 'theta3': theta3,
                  'bias1': bias1, 'bias2': bias2, 'bias3': bias3}
    return parameters


def test_run(parameters):
    while True:
        data_vector = process_image().flatten()

        layer2 = tf.sigmoid(tf.matmul(data_vector, parameters['theta1']) + parameters['bias1'])
        layer3 = tf.sigmoid(tf.matmul(layer2, parameters['theta2']) + parameters['bias2'])
        hypothesis = tf.sigmoid(tf.matmul(layer3, parameters['theta3']) + parameters['bias3'])
        prediction = tf.argmax(hypothesis, axis=1)
        with tf.Session() as sess:
            prediction = sess.run(prediction)[0]


train_data()
