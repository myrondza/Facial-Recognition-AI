import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
videoObjcap = cv2.VideoCapture(0)


while(True):
    cap , frame = videoObjcap.read()
    scale = 10
    width = int(frame.shape[1]*20/100)
    height = int(frame.shape[0]*20/100)
    dim = (width,height)
    resized = cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor= 1.1,minNeighbors= 5,minSize=(10, 10)) 
    for (x, y, w, h) in faces:
        cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 255, 0), 2)
    cv2.imshow('Capturing Frames',resized)
    
    if(cv2.waitKey(1) & 0xFF == ord('s')):
        crop = resized[y:y+h, x:x+w]
        cv2.imwrite('facechk.jpg',crop)
        videoObjcap.release()
        print("Success")
cv2.destroyAllWindows()