import os

# import numpy as np
# import sys
# from PIL import  Image
import cv2
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('train/trainer.yml')
#准备识别的图片
imgg=cv2.imread('y4.jpg',0)
gray=cv2.cvtColor(imgg,cv2.COLOR_BAYER_BG2GRAY)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
faces=face_detector.detectMultiScale(imgg, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, b) in faces:
    img = cv2.rectangle(imgg, (x, y), (x + w, y + b), (255, 255, 0), 2) #
    #人脸识别
    id,confidence=recognizer.predict(gray[y:y+b,x:x+w])
    print('标签id:',id,'置信评分:',confidence)
cv2.imshow('result',imgg)
cv2.waitKey(0)
cv2.destroyAllWindows()



