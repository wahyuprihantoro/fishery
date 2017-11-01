import PIL
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

labels = pd.read_csv('label.csv')['label'].values
le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
labels = np.eye(10)[labels]
labels = labels[:3500]
print(labels.shape)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def load_image(infilename):
    img = np.asarray(PIL.Image.open(infilename))
    return rgb2gray(img)


path = 'dataset/new_train/'
images = []
for i in range(1, 3501):
    if i % 1000 == 0:
        print(str(i))
    images += [load_image(path + str(i) + ".jpg")]

import os

path = os.getcwd() + '/dataset/test_stg1/'
filenames = []
for file in os.listdir(path):
    if file.endswith('.jpg'):
        filenames += [path + '/' + file]
test_images = []

for fn in filenames:
    test_images += [load_image(fn)]

n_classes = 10
batch_size = 100

x = tf.placeholder('float', shape=[None, 32, 32, 1])
y = tf.placeholder('float', shape=[None, 10])

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([8 * 8 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    # x = tf.reshape(x, shape=[-1, 32, 32, 1])
    # print(x.shape)
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    print(conv2.shape)
    fc = tf.reshape(conv2, [-1, 8 * 8 * 64])
    print(fc.shape)
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)
    print(fc.shape)

    output = tf.matmul(fc, weights['out']) + biases['out']
    print(output.shape)

    return output


def train_neural_network(x):
    global accuracy
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 2
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            i = 0
            epoch_loss = 0
            while i < len(images):
                start = i
                end = i + batch_size
                batch_x = np.array(images[start:end])
                batch_y = np.array(labels[start:end])
                batch_x = batch_x.reshape(-1, 32, 32, 1)
                batch_y = batch_y.reshape(-1, 10)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            img = np.array(images).reshape(3500, 32, 32, 1)
            print('Accuracy:', accuracy.eval({x: img, y: labels}))
        pred = tf.arg_max(prediction, 1)
        print(pred.eval(feed_dict={x: test_images}))


train_neural_network(x)
