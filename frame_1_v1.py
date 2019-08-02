# 框架1
# 框架1使用tf.layers来构成神经网络结构，使用tf.Session()来执行训练过程
# 框架1利用函数来实现模块化
# 框架1以一个简单的cnn演示该框架为例子演示该框架的结构和使用，该cnn由两层卷基层两层池化层，一层全连接层构成。

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
learning_rate = 0.001
num_steps = 8000
batch_size = 128
display_step = 100

# 网络常数
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.5  # Dropout, probability to keep units

# 输入变量proje
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
tf_is_training = tf.placeholder(tf.bool, None)  # 对dropout是否使用的控制


# 定义功能函数:

# 准确率函数
def compute_accuracy(sess0, out, v_xs, v_ys):
    """
    这个仅适用于没有dropout，即只适用于仅适用Tensorflow变量X和Y的
    """
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
    accuracy0 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess0.run(accuracy0, feed_dict={X: v_xs, Y: v_ys})
    return result


# 显示图片
def mnist_display():
    pass
# plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
# plt.title('%i' % np.argmax(mnist.train.labels[0]))
# plt.show()


# 定义网络结构：

# CNN简单结构:两个卷积层，两个池化层，加一个全连接层
def simple_cnn(tf_x):
    image = tf.reshape(tf_x, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
        inputs=image,  # shape (28, 28, 1)
        filters=16,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )  # -> (28, 28, 16)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=2,
        strides=2,  # 由于池化strides取2才能保证不重复的框住输入矩阵的每个小块
    )  # -> (14, 14, 16)
    conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)  # -> (14, 14, 32)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # -> (7, 7, 32)
    flat = tf.contrib.layers.flatten(pool2) # -> (7*7*32, )
    output = tf.layers.dense(flat, 10)  # output layer
    # 这里没有再调用softmax激活函数，因为代价函数自己会做softmax，又不用输出层就能得出分类情况，因此就没有再调用softmax
    return output


# 定义训练过程：

outputs = simple_cnn(X)
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=outputs)  # compute cost
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


# 执行神经网络：
with tf.Session() as sess:
    # 开始训练：
    sess.run(init_op)
    for step in range(1, num_steps+1):
        b_x, b_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={X: b_x, Y: b_y})
        if step % display_step == 0 or step == 1:
            print('Epoch:', step, 'train:', compute_accuracy(sess, outputs, test_train_x, test_train_y),
                  ' | test:', compute_accuracy(sess, outputs, test_x, test_y))
    print('******************************************************************************************')
    # 训练结束：
    print('End:')
    print('train:', compute_accuracy(sess, outputs, test_train_x, test_train_y),
          ' | test:', compute_accuracy(sess, outputs, test_x, test_y))
    # 分类图片测试
    test_output = sess.run(outputs, {X: test_x[:10]})
    pred_y = np.argmax(test_output, 1)
    print(pred_y, 'prediction number')
    print(np.argmax(test_y[:10], 1), 'real number')


# 备用实现：


# 1.将训练过程的定义函数化
# def train_simple_cnn(tf_y, out, l_r):
#     loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=out)  # compute cost
#     train_op = tf.train.AdamOptimizer(l_r).minimize(loss)
#     return train_op
# train = train_simple_cnn(Y, outputs, learning_rate)

# 2.加入tf.metrics.accuracy，但注意它的特殊用法，如果想计算所有数据的一个准确率要放到最后执行一次
# accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
#     labels=tf.argmax(Y, axis=1), predictions=tf.argmax(outputs, axis=1),)[1]
# ac = sess.run(accuracy, feed_dict={X: test_x, Y: test_y})
# print('test accuracy: %.4f' % ac)

# 3.softmax的交叉熵代价函数另一种写法：
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=output, labels=Y))  # 使用tf.nn.softmax_cross_entropy_with_logits这个函数的时候,logits参数传递的应该是输出层的z（使用softmax之前输出）

# 4.关于dropout
# 这里的dropout是用layers实现的，还可以用tf.nn.dropout(outputs, keep_prob)来实现

# 5. 关于展开进入全连接层之前的展开的另一种实现：不用flatten函数，而是计算形状再用reshape
# flat = tf.reshape(pool2, [-1, 7 * 7 * 32])  # -> (7*7*32, )
