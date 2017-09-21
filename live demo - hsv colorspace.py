import cv2
import numpy as np

cap = cv2.VideoCapture('/home/stephen/Desktop/source.avi')


wait_time = 100

def nothing(arg): pass

#takes image and range
#returns parts of image in range
def only_color(img, (h,s,v,h1,s1,v1)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([h,s,v]), np.array([h1,s1,v1])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res, mask

#create trackbars for user input
cv2.namedWindow('image')
cv2.createTrackbar('h','image',40,255,nothing)
cv2.createTrackbar('s','image',0,255,nothing)
cv2.createTrackbar('v','image',0,255,nothing)
cv2.createTrackbar('h1','image',60,255,nothing)
cv2.createTrackbar('s1','image',255,255,nothing)
cv2.createTrackbar('v1','image',255,255,nothing)

while True:
    _, img = cap.read()
    h=cv2.getTrackbarPos('h','image')
    s=cv2.getTrackbarPos('s','image')
    v=cv2.getTrackbarPos('v','image')
    h1=cv2.getTrackbarPos('h1','image')
    s1=cv2.getTrackbarPos('s1','image')
    v1=cv2.getTrackbarPos('v1','image')

    img, mask = only_color(img, (h,s,v,h1,s1,v1))

    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    k = cv2.waitKey(wait_time)
    if k==27: break
    
cap.release()
cv2.destroyAllWindows()
