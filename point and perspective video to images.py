import cv2
cap = cv2.VideoCapture('/home/stephen/Desktop/chromakey.avi')
images = []
frame_number = 0
for i in range(frame_number): _,_ = cap.read()
for i in range(2000):
    _, img = cap.read()
    try: l = img.shape
    except: break
    images.append(img)
    #cv2.imshow('img', img)
    #import time
    cv2.imwrite('/home/stephen/Desktop/images/'+str(frame_number)+'.png', img)
    frame_number += 1
    if frame_number %50==0: print frame_number
    #k = cv2.waitKey(1)
    #if k==27:break
print frame_number    
cv2.destroyAllWindows()
cap.release()
