# --*coding -- :utf-8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

#设定参数
learning_rate=0.01 #学习率
training_epochs=1000#训练数量
display_step=50#每50个走一波吧就

#训练数据
train_x=numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_y=numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])


#设置placeholder
X=tf.placeholder("float")
Y=tf.placeholder("float")

#设置模型的权重和偏置(偏见bias）

W=tf.Variable(rng.randn(),name="weight")
b=tf.Variable(rng.randn(),name="bias")

#设置线性回归的方程
pred= tf.add(tf.multiply(X,W),b)

#设置cost为均方差
cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*17)
#梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#初始化所有variables
init = tf.global_variables_initializer()


#训练
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for(x,y) in zip(train_x,train_y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
            # if (epoch+1)%display_step==0:
            #             #  c= sess.run(cost,feed_dict={X:train_x,Y:train_y})
            # print("Epoch:",'%04d' % (epoch+1),"cost=","{:,9f}".format(c)
            # "W=",sess.run((W)),"b=",sess.run(b))
            #
            # print"Finished!"
            # training_cost=sess.run(cost,feed_dict={X:train_x,Y:train_y})
            # print"Training cost=",training_cost,"W=",sess.run(W),"b=",sess.run(b)

        #做图
        print('w=',sess.run(W))
        print('b=',sess.run(b))
        plt.plot(train_x,train_y,'ro',label='Original data')
        plt.plot(train_x,sess.run(W)*train_x+sess.run(b),label='Fitted line')
        plt.legend()
        plt.show()





