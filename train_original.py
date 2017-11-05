# image -> ten categories
# By Fang Wan

import tensorflow as tf
import math
import os
from data_prep_npz import read_data_sets
tf.set_random_seed(0)
import numpy as np
import pandas as pd

mnist = read_data_sets("data")

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 200, 200, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.int64, [None])
# variable learning rate
lr = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 3, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)


W4 = tf.Variable(tf.truncated_normal([50 * 50 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7 50x50
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 50 * 50 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), Y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

# Setting up the indicators
checkpoint_path = './checkpoint'
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# You can call this function in a loop to train the model, 100 images at a time
a_train = []
a_test = []
test_prediction = []
for i in range(10001):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    batch_Y = batch_Y.reshape([100])
    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate})
    # compute training values for visualisation
    if i%10==0:
        a = sess.run(accuracy, {X: batch_X, Y_: batch_Y})
        print(str(i) + ": accuracy:" + str(a) + " (lr:" + str(learning_rate) + ")")
        a_train.append(a)
    # compute test values for visualisation
    if i%100==0:
        a = sess.run(accuracy, {X: mnist.test.images, Y_: mnist.test.labels.reshape([1000])})
        a_test.append(a)
        y = sess.run(Y, {X: mnist.test.images})
        test_prediction.append(y)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a))

saver.save(sess, checkpoint_path + '/Network')

print("max test accuracy: " + str(max(a_test)))
np.savez(checkpoint_path+'/accuracy', a_test=a_test, a_train=a_train, test_prediction=test_prediction)

# max test accuracy: 0.98 in 10000 steps
