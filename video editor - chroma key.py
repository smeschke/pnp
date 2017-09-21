import cv2
import numpy as np
cap = cv2.VideoCapture('/home/stephen/Desktop/source.avi')
#cap = cv2.VideoCapture(1)
fps = 30.0
cols, rows = 720,780
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/stephen/Desktop/chromakey.avi', fourcc, fps, (cols, rows))
outmask = cv2.VideoWriter('/home/stephen/Desktop/chromakeymask.avi', fourcc, fps, (cols, rows))


def only_color(img, (h,s,v,h1,s1,v1)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([h,s,v]), np.array([h1,s1,v1])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((15,15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res, mask

#range for the chroma key (green)
h,s,v,h1,s1,v1 = 8,0,101,151,69,240
h,s,v,h1,s1,v1 = 40,100,80,60,168,188
h,s,v,h1,s1,v1 = 40,0,0,60,255,255

while True:
    _, img = cap.read()
    try: l = img.shape
    except: break
    green, green_mask = only_color(img, (h,s,v,h1,s1,v1))
    not_green = cv2.bitwise_and(img, img, mask=255-green_mask)
    
    y = 0
    y1 = 700
    x = 0
    x1 =720
    
    roi = not_green[y:y1, x:x1].copy()
    not_green = np.zeros_like(img)
    not_green[y:y1, x:x1] = roi
    cv2.imshow('mask', green_mask)
    cv2.imshow('img', not_green)
    
    mask = np.zeros_like(img)
    mask[:,:,0] = green_mask
    mask[:,:,1] = green_mask
    mask[:,:,2] = green_mask
    
    mask_roi = mask[y:y1, x:x1].copy()
    bg = np.zeros_like(img)
    bg[y:y1, x:x1] = mask_roi
    


    
    
    k=cv2.waitKey(1)
    if k==27: break
    out.write(not_green)
    outmask.write(mask)
cv2.destroyAllWindows()
