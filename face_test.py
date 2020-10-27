import os
import cv2
import numpy as np
import sys
from PIL import  Image

recognizer=cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
def getImagesAndLables(path):
    imagePaths=[os.path.join(path,f)for f in os.listdir(path)]
    faceSamples=[]
    ids=[]
    #遍历列表中的图片
    for imagePath in imagePaths:
        #打开图片
        PIL_img=Image.open(imagePath).convert('L')#convert
        #将图像转化为数组
        img_numpy=np.array(PIL_img,'uint8')
        #获取每张图片的id
        id=int(os.path.split(imagePath)[-1].split(".")[0])
        print(id)
        faces=detector.detectMultiScale(img_numpy)
        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])#把人脸放到faceSamples里
            ids.append(id)
    return faceSamples,ids


path='./data1'
faces,ids=getImagesAndLables(path)
recognizer.train(faces,np.array(ids))
recognizer.write('train/trainer.yml')





