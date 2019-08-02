# 基于框架1的一个DNN实现
# 该DNN结构：
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

# 输入变量
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)


# 定义功能函数:

# 准确率函数
def compute_accuracy(sess0, out, v_xs, v_ys):
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
    accuracy0 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess0.run(accuracy0, feed_dict={X: v_xs, Y: v_ys})
    return result


# 定义网络结构：

# simple_dnn是一个只有一层隐藏层的最简单的结构：
def simple_dnn(x):
    l1 = tf.layers.dense(x, 256, tf.nn.relu)
    l2 = tf.layers.dense(l1, 256, tf.nn.relu)
    output = tf.layers.dense(l2, 10)
    # 这里没有再调用softmax激活函数，因为代价函数自己会做softmax，又不用输出层就能得出分类情况，因此就没有再调用softmax
    return output


# 定义训练过程：

outputs = simple_dnn(X)
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