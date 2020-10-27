import cv2

def face_rec_demo():

    face_rec=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    faces=face_rec.detectMultiScale()
    for x,y,w,b in faces:
        img=cv2.rectangle(img,(x,y),(x+w,y+b),(255,255,0),thickness=2)
    cv2.imshow('result',img)

img=cv2.imread('C:/Users/10647/Desktop/demo1/timg (4).jfif')
face_rec_demo()
cv2.waitKey(0)
cv2.destroyAllWindows()