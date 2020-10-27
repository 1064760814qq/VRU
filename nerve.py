import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

# matrix1 = tf.constant([[3,3]])
# matrix2 = tf.constant([[2],[2]])
# product=tf.matmul(matrix1,matrix2)
#
# with tf.Session() as sess:
#     result=sess.run(product)
#     print(result)

state=tf.Variable(0,name='OK')
# print(state.name)
one=tf.constant(1)
new=tf.add(state,one)

update=tf.assign(state,new)
print(update)






