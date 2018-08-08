import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成500个随机点
x_data = np.linspace(-4, 4, 500)[:, np.newaxis]
noise = np.random.normal(0, 1, x_data.shape)
y_data = np.square(x_data) + noise - 25

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络的中间层
weights_l1 = tf.Variable(tf.random_normal([1, 10]))
biases_l1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_l1 = tf.matmul(x, weights_l1) + biases_l1
l1 = tf.nn.tanh(Wx_plus_b_l1)

# 定义神经网络输出层
weights_l2 = tf.Variable(tf.random_normal([10, 1]))
biases_l2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_l2 = tf.matmul(l1, weights_l2) + biases_l2
prediction = tf.nn.tanh(Wx_plus_b_l2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#定义会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1300):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    # 获取预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.plot(x_data, prediction_value, 'b-', lw=1)
    plt.show()
