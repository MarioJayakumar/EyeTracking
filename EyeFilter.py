import numpy as np
import cv2

cap = cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
glasses = cv2.imread("aviator.png")

eyeTotal = []

while(True):
    
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30,30))
    

    for (x,y,w,h) in faces:
        cut = frame[y:y+h, x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0))
        eyeGet=eye_cascade.detectMultiScale(cv2.cvtColor(cut,cv2.COLOR_BGR2GRAY),
            scaleFactor = 1.1, minNeighbors=20,
            minSize=(30,30)) + (x,y,0,0)
        eyeTotal= eyeGet
        
    circleTotal = []
    if isinstance(eyeTotal,np.ndarray):
        count = 0
        for (x,y,w,h) in eyeTotal:
            cut = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(cut,cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(gray,10,40)
            canny = cv2.GaussianBlur(canny,(13,13),4)
            result = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT,1,20,
                param1=100,param2=25,minRadius=0,maxRadius=50)
            if isinstance(result,np.ndarray):
                circleTotal.extend(result[0].tolist())
                print(circleTotal)
                # if len(circleTotal) == 2:
                #     while (True):
                #         count = 0
                        

           
            count += 1
            for i in circleTotal:
                i[0] += x
                i[1] += y


    
    for j in circleTotal:
        cv2.circle(frame,(int(j[0]),int(j[1])),int(j[2]),(0,0,255),-2)

 
    #cannot iterate if singular value
    if isinstance(eyeTotal,np.ndarray):
        for (x,y,w,h) in eyeTotal:
    	    cv2.rectangle(frame,(x,y), (x+w,y+h),(255,0,0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('p'):
        while(True):
            if cv2.waitKey(1) & 0xFF == ord('p'):
                break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#two rects are similar if their perimeters are within 10% of each other
def isSimilar(rect1, rect2):
    peri1 = 2*(rect1.w + rect1.h)
    peri2 = 2*(rect1.w + rect2.h)

    isSim = false
    if peri1 > peri2:
        isSim = peri2/peri1 < 10
    else:
        isSim = peri1/peri2 < 10

    return isSim

#sees if two rects centers' are close to each other. 20 px radius is close
def isClose(rect1, rect2):

    return abs(rect1.x-rect2.x) < 20 and abs(rect1.y-rect2.y) < 20


    
