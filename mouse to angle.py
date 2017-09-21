import cv2
import numpy as np
import math
import os
#mouse callback function
click_list, coords = [(100,100)],(0,0)
global click_list, coords
def callback(event,x,y,flags,param):
    global click_list, coords
    if event==cv2.EVENT_LBUTTONDOWN:
        click_list.append((x,y))
    coords = x,y
cv2.namedWindow('img')
cv2.setMouseCallback('img', callback)
#distance function
def distance(a,b): return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
def get_slope(a,b):
    try: return (float(a[1])-b[1])/(a[0]-b[0])
    except: return -1

#law of cos function for finding angles
def get_angle(A,B,C):
    a = distance(B, C)
    b = distance(A, C)
    c = distance(A, B)
    top = a**2 + b**2 - c**2
    bottom = 2*a*b
    return round(math.acos(top/bottom) * 360 / (2*math.pi))

while True:
    #grab image
    img = np.zeros((600,600,3),np.uint8)
    cv2.line(img, click_list[-1], coords, (123,123,123), 4)
    a,b =click_list[-1], coords
    slope = get_slope(a,b)
    t =  math.atan(-slope)
    if a[0]>b[0]:
        print 270-360*t/6.24
    else:
        print 90-360*t/6.24
    
    
    cv2.imshow('img',img)
    k = cv2.waitKey(1)
    if k==27: break
cv2.destroyAllWindows()
