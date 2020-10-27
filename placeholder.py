import tensorflow as tf


# tf.reset_default_graph()
logdir='D:/log'

a=tf.constant(100,name='a')
b=tf.constant(5,name='b')
c=tf.add(a,b,name='c')

writer=tf.summary.FileWriter(logdir,tf.get_default_graph())
writer.close()



