# 外部库
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# import matplotlib.pyplot as plt

# 导入数据集
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
test_x = mnist.test.images
test_y = mnist.test.labels
test_train_x = mnist.train.images[:10000]
test_train_y = mnist.train.labels[:10000]

# 数据集信息
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

# 超参数
learning_rate = 0.0005 # 0.0005 0.003 0.0007
num_steps = 9000
batch_size = 128
display_step = 100

# 网络常数
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.6  # 对于layers.dropout来说这是丢弃率
lam = 0.0001
# 0.0005  0.0005 0.00005

# 输入变量
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
tf_is_training = tf.placeholder(tf.bool, None)  # 对dropout是否使用的控制


# 定义功能函数:

# 准确率函数
def compute_accuracy(sess0, out, v_xs, v_ys):
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
    accuracy0 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess0.run(accuracy0, feed_dict={X: v_xs, Y: v_ys, tf_is_training: False})
    return result


# 显示图片
def mnist_display():
    pass
# plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
# plt.title('%i' % np.argmax(mnist.train.labels[0]))
# plt.show()


# 定义网络结构：

# 结构说明：
# 这是基于LetNet改进的CNN结构,两个卷积层两个池化层，之后跟两个全连接层，全连接层都dropout,此外每一层都加入了L2正则化。

def cnn3(tf_x, dp, is_training, lam0):
    image = tf.reshape(tf_x, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
        inputs=image,  # shape (28, 28, 1)
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0)
    )  # -> (28, 28, 16)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=2,
        strides=2,  # 由于池化strides取2才能保证不重复的框住输入矩阵的每个小块
    )  # -> (14, 14, 16)
    conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))  # -> (14, 14, 32)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # -> (7, 7, 32)
    fc = tf.contrib.layers.flatten(pool2)
    h1 = tf.layers.dense(fc, 256, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    d1 = tf.layers.dropout(h1, rate=dp, training=is_training)
    h2 = tf.layers.dense(d1, 256, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))
    d2 = tf.layers.dropout(h2, rate=dp, training=is_training)
    output = tf.layers.dense(d2, 10, kernel_regularizer=tf.contrib.layers.l2_regularizer(lam0))  # output layer
    # 这里没有再调用softmax激活函数，因为代价函数自己会做softmax，又不用输出层就能得出分类情况，因此就没有再调用softmax
    return output


# 定义训练过程：

outputs = cnn3(X, dropout, tf_is_training, lam)
loss0 = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=outputs)  # compute cost
loss1 = tf.losses.get_regularization_loss()
loss = loss0 + loss1
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


# 执行神经网络：
with tf.Session() as sess:
    # 开始训练：
    sess.run(init_op)
    for step in range(1, num_steps+1):
        b_x, b_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={X: b_x, Y: b_y, tf_is_training: True})
        if step % display_step == 0 or step == 1:
            print('Epoch:', step, 'train:', compute_accuracy(sess, outputs, test_train_x, test_train_y),
                  ' | test:', compute_accuracy(sess, outputs, test_x, test_y))
    print('******************************************************************************************')
    # 训练结束：
    print('End:')
    print('train:', compute_accuracy(sess, outputs, test_train_x, test_train_y),
          ' | test:', compute_accuracy(sess, outputs, test_x, test_y))
    # 分类图片测试
    test_output = sess.run(outputs, {X: test_x[:10], tf_is_training: False})
    pred_y = np.argmax(test_output, 1)
    print(pred_y, 'prediction number')
    print(np.argmax(test_y[:10], 1), 'real number')