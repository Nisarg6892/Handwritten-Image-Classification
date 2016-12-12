import tensorflow as tf
import numpy as np
# import preprocess_in_memory as ppim
import preprocess

# Parameters

learning_rate = 0.001
batch_size = 20

im_w = 128
im_h = 128
total_iterations = 50
# Network Parameters
n_input = im_w * im_h  # MNIST data input (img shape: 28*28)
n_classes = 2  # MNIST total classes (0-9 digits)
dropout = 0.60  # Dropout, probability to keep units

# tf Graph input
x_1 = tf.placeholder(tf.float32, [None, im_w, im_h])
x_2 = tf.placeholder(tf.float32, [None, im_w, im_h])
y = tf.placeholder(tf.int32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    # print x.shape()
    # tf.Print(x,"testing:")
    print x.get_shape()
    print W.get_shape()
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, im_w, im_h, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # output x: 1650 y: 1274

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # output x: 825 y : 637
    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2)

    return conv4


def final_model(image_1_x, image_2_x, weights_x, weights_y, biases_x, biases_y, dropout):
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input

    image_1_conv_output = conv_net(image_1_x, weights_x, biases_x)
    image_2_conv_output = conv_net(image_2_x, weights_y, biases_y)

    image_1_conv_output = tf.reshape(image_1_conv_output, [-1, weights['combined_wd1'].get_shape().as_list()[0] / 2])
    image_2_conv_output = tf.reshape(image_2_conv_output, [-1, weights['combined_wd1'].get_shape().as_list()[0] / 2])
    combined_conv_output = tf.concat(1, [image_1_conv_output, image_2_conv_output])
    # image_2_conv_output = image_1_conv_output

    # fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(combined_conv_output, weights['combined_wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)

    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.relu(fc3)

    fc4 = tf.add(tf.matmul(fc3, weights['wd4']), biases['bd4'])
    fc4 = tf.nn.relu(fc4)

    # Apply Dropout
    fc4 = tf.nn.dropout(fc4, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc4, weights['out']), biases['out'])
    return out


# Store layers weight & bias
input_channels = {
    'wc1': 1, 'wc2': 32, 'wc3': 64, 'wc4': 128
}

weights_x = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, input_channels['wc1'], input_channels['wc2']])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, input_channels['wc2'], input_channels['wc3']])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wc3': tf.Variable(tf.random_normal([5, 5, input_channels['wc3'], input_channels['wc4']])),
    'wc4': tf.Variable(tf.random_normal([5, 5, input_channels['wc4'], 2 * input_channels['wc4']])),

}
weights_y = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, input_channels['wc1'], input_channels['wc2']])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, input_channels['wc2'], input_channels['wc3']])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wc3': tf.Variable(tf.random_normal([5, 5, input_channels['wc3'], input_channels['wc4']])),
    'wc4': tf.Variable(tf.random_normal([5, 5, input_channels['wc4'], 2 * input_channels['wc4']])),
}
weights = {
    # combined weight
    'combined_wd1': tf.Variable(tf.random_normal([2 * (im_w / 16) * (im_h / 16) * (2 * input_channels['wc4']), 256])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd2': tf.Variable(tf.random_normal([256, 128])),
    'wd3': tf.Variable(tf.random_normal([128, 64])),
    'wd4': tf.Variable(tf.random_normal([64, 32])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([32, n_classes]))
}

biases_x = {
    'bc1': tf.Variable(tf.random_normal([input_channels['wc2']])),
    'bc2': tf.Variable(tf.random_normal([input_channels['wc3']])),
    'bc3': tf.Variable(tf.random_normal([input_channels['wc4']])),
    'bc4': tf.Variable(tf.random_normal([2 * input_channels['wc4']])),

}
biases_y = {
    'bc1': tf.Variable(tf.random_normal([input_channels['wc2']])),
    'bc2': tf.Variable(tf.random_normal([input_channels['wc3']])),
    'bc3': tf.Variable(tf.random_normal([input_channels['wc4']])),
    'bc4': tf.Variable(tf.random_normal([2 * input_channels['wc4']])),

}
biases = {
    'bd1': tf.Variable(tf.random_normal([256])),
    'bd2': tf.Variable(tf.random_normal([128])),
    'bd3': tf.Variable(tf.random_normal([64])),
    'bd4': tf.Variable(tf.random_normal([32])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# with tf.device('/cpu:0'):

# Construct model
# pred = conv_net(x, weights, biases, keep_prob)
pred = final_model(x_1, x_2, weights_x, weights_y, biases_x, biases_y, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# train_images = np.load('train_pairs.npy')
# train_labels = np.load('train_labels.npy')
# a, b = train_images.shape
# train_images = np.reshape(train_images, (a, 2, b / 2))
# print train_labels.shape
# # Launch the graph
# with tf.Session() as sess:
#     sess.run(init)
#     step = 1
#     # Keep training until reach max iterations
#     while step * batch_size < training_iters:
#
#         train_images_x = train_images[count:count + batch_size, 0, :]
#         train_images_y = train_images[count:count + batch_size, 1, :]
#         sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
#                                        keep_prob: dropout})
#         if step % display_step == 0:
#             # Calculate batch loss and accuracy
#             loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
#                                                               y: batch_y,
#                                                               keep_prob: 1.})
#             print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
#                   "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                   "{:.5f}".format(acc))
#         step += 1
#     print("Optimization Finished!")
#
#     # Calculate accuracy for 256 mnist test images
#     print("Testing Accuracy:", \
#           sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
#                                         y: mnist.test.labels[:256],
#                                         keep_prob: 1.}))
#

saver = tf.train.Saver()
total_images = preprocess.total_images
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, 'my-cnn3')
    #testing part
    test_images = np.load('test_images.npy')
    test_labels = np.load('test_labels.npy')
    total_acc = 0.0
    for i in range(97):
        x_batch = test_images[i*20:(i+1)*20, 0, :, :]
        y_batch = test_images[i*20:(i+1)*20, 1, :, :]
        label_batch = test_labels[i*20:(i+1)*20, :]
        total_acc += accuracy.eval(feed_dict={x_1: x_batch,
                                                       x_2: y_batch,
                                                       y:label_batch ,keep_prob:dropout})

    print total_acc/97


"""
    #training part
    batch_count = 0
    epoch_count = 0
    loopControl = True
    while (loopControl):
        if (batch_size * (batch_count + 1) >= total_images):
            temp_batch_size = total_images - batch_size * (batch_count)
            print temp_batch_size
            print batch_count
            x_batch, label_batch = preprocess.get_batch(batch_count, temp_batch_size)
            x_1_batch = x_batch[:, 0, :, :]
            x_2_batch = x_batch[:, 1, :, :]
            batch_count = 0
            epoch_count += 1
            print epoch_count
            if epoch_count == total_iterations:
                loopControl = False
            sess.run([optimizer, cost], feed_dict={x_1: x_1_batch,
                                                   x_2: x_2_batch,
                                                   y: label_batch, keep_prob: dropout})

            saver.save(sess, 'my-cnn' + str(epoch_count))

        else:
            print "Training started" + " Batch Count: " + str(batch_count) + " Epoch Count: " + str(epoch_count)
            x_batch, label_batch = preprocess.get_batch(batch_count, batch_size)
            x_1_batch = x_batch[:, 0, :, :]
            x_2_batch = x_batch[:, 1, :, :]

            sess.run([optimizer, cost], feed_dict={x_1: x_1_batch,
                                                   x_2: x_2_batch,
                                                   y: label_batch, keep_prob: dropout})

            batch_count += 1
    saver.save(sess, 'my-cnn-final')
"""
