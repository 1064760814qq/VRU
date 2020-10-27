import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#1.导入库
import cv2

#2.加载图片
image = cv2.imread('timg (1).jfif')
#3.创建窗口
cv2.namedWindow('mingming')

#4.显示图片
cv2.imshow('minghao',image)
#5.暂停窗口
cv2.waitKey(0)
#6.关闭窗口
cv2.destroyAllWindows()