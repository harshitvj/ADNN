import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Defining Parameters
total_epochs = 500
learning_rate = 0.06

input_layer_nodes = 2208
hidden1_layer_nodes = 500
hidden2_layer_nodes = 500
hidden3_layer_nodes = 500
output_layer_nodes = 7

cost_history = np.empty(shape=[0], dtype=np.float32)
model_path = "./Model"
filename = 'balanced_data.npy'

training_data = list(np.load(filename))
x_data = [x[0] for x in training_data]
y_data = [x[1] for x in training_data]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)


# Splitting 80/20 of the data to be train/test data
train_X, test_X, train_Y, test_Y = train_test_split(x_data, y_data, test_size=0.20, random_state=675)

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

hypothesis = multi_layer_perceptron(X, weights, biases)

# defining cost
cost_funtion = tf.reduce_mean(tf.square(hypothesis - Y))

# defining Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_funtion)

sess = tf.Session()
sess.run(init)
writer = tf.summary.FileWriter('./logs', sess.graph)

mse_history = []
accuracy_history = []

for epoch in range(total_epochs):
    cost = sess.run([cost_funtion, optimizer], feed_dict={X: train_X, Y: train_Y})[0]
    cost_history = np.append(cost_history, cost)

    correct_prediction = tf.equal(tf.argmax(hypothesis, axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    prediction_y = sess.run(hypothesis, feed_dict={X: train_X})
    mse = tf.reduce_mean(tf.square(prediction_y - train_Y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)

    accuracy = sess.run(accuracy, feed_dict={X: train_X, Y: train_Y})
    accuracy_history.append(accuracy)

    print('Epoch:', epoch, 'Cost:', cost, 'MSE:', mse_, 'Accuracy:', accuracy * 100, '%')

save_path = saver.save(sess, model_path)
print("Model saved in {}".format(save_path))

np.save('mse_data.npy', mse_history)
np.save('accuracy_data.npy', accuracy_history)

plt.plot(mse_history, 'r-')
plt.show()
plt.plot(accuracy_history, 'b-')
plt.show()
sess.close()
