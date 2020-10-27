import cv2

#加载人脸模型库
face_engine = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
camera=cv2.VideoCapture(0)
i=1
# 1.设置一个窗口
cv2.namedWindow('shuoshuo')
# 2.如果正确的话，
while True:
    # 2.1读取摄像头的帧画面
    # 2.2显示图片
    # 2.3暂停窗口
    # 3.释放资源
    # 4.关闭窗口
    ret,frame=camera.read()
    faces = face_engine.detectMultiScale(frame,scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,b) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + b), (255, 255, 0), 2)
    cv2.imshow('shuo',frame)
    if cv2.waitKey(5) & i==2:
        break
camera.release()
cv2.destroyAllWindows()






