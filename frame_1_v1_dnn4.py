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
learning_rate = 0.0005
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
tf_is_training = tf.placeholder(tf.bool, None)  # 对dropout是否使用的控制


# 定义功能函数:

# 准确率函数
def compute_accuracy(sess0, out, v_xs, v_ys):
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
    accuracy0 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess0.run(accuracy0, feed_dict={X: v_xs, Y: v_ys, tf_is_training: False})
    return result


# 定义网络结构：

# 这个神经网络的结构是由两层隐藏层组成，特点是每层隐藏层都加入了dropout
# 这里的dropout是用layers实现的，还可以用tf.nn.dropout(outputs, keep_prob)来实现
def dnn4(x, dp, is_training):
    h1 = tf.layers.dense(x, 256, tf.nn.relu)
    d1 = tf.layers.dropout(h1, rate=dp, training=is_training)
    h2 = tf.layers.dense(d1, 256, tf.nn.relu)
    d2 = tf.layers.dropout(h2, rate=dp, training=is_training)
    output = tf.layers.dense(d2, 10)
    return output


# 定义训练过程：

outputs = dnn4(X, dropout, tf_is_training)
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=outputs)  # compute cost
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
