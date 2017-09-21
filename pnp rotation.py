import cv2
import numpy as np
import math
import os
import pandas as pd

#captrue webcam
cap = cv2.VideoCapture('/home/stephen/Desktop/pose.MP4')
fps = 30.0
cols, rows = 1280,720
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/stephen/Desktop/pnp_out.avi', fourcc, fps, (cols, rows))
num_images  = len(os.listdir('/home/stephen/Desktop/images/'))

#get data
files = []
files.append('/home/stephen/Desktop/six.csv')
files.append('/home/stephen/Desktop/three.csv')
files.append('/home/stephen/Desktop/twelve.csv')
files.append('/home/stephen/Desktop/nine.csv')
data = []
for f in files:
    df = pd.read_csv(f)
    data.append(df.values)
source_data = zip(data[0],data[1],data[2],data[3])

#get data
files = []
files.append('/home/stephen/Desktop/posesix.csv')
files.append('/home/stephen/Desktop/posethree.csv')
files.append('/home/stephen/Desktop/posetwelve.csv')
files.append('/home/stephen/Desktop/posenine.csv')
data = []
for f in files:
    df = pd.read_csv(f)
    data.append(df.values)
pose_data = zip(data[0],data[1],data[2],data[3])


def get_slope(a,b):
    try: return (float(a[1])-b[1])/(a[0]-b[0])
    except: return 0
    
def distance(a,b): return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

#takes the source image, the target, and the soruce data
#returns a warped image of the soruce fits the target
def warp(img, target, source, near, shape):
    values = []
    d = []
    for a,b in zip(target, source): d.append((a,b))
    target, source = zip(*d)
    M = cv2.getPerspectiveTransform(np.array(source, np.float32), np.array(target, np.float32))
    return cv2.warpPerspective(img, M, (shape[1], shape[0]))

frame_number = 0
while True:
    #grab image
    _, img = cap.read()

    #get target in source (white board)
    target = pose_data[frame_number]
    frame_number+=1
    a,b = target[0], target[2]
    t =  math.atan(-get_slope(a,b))
    if a[0]>b[0]: rotation = .75-t/(2*math.pi)
    else: rotation = .25-t/(2*math.pi)
    #print rotation
    #cv2.circle(img, (a[0],a[1]), 5, (123,123,123), 5)
    #cv2.circle(img, (b[0],b[1]), 5, (123,123,123), 5)
    near = int(num_images*rotation)
    if near>len(source_data)-1:
        print 'overflow', near, rotation, num_images*rotation, num_images
        near = 0
        
    pose_coordinates = source_data[near]
    
    
    #nearest matching image
    pnp = cv2.imread('/home/stephen/Desktop/images/'+str(near)+'.png')
    mask = cv2.imread('/home/stephen/Desktop/masks/'+str(near)+'.png',0)
    #for point in pose_coordinates: cv2.circle(pnp, (point[0],point[1]), 15, (255,255,255), 2)
    print near
    
    #warp image and mask
    img2 = warp(pnp, target, pose_coordinates, near, img.shape)
    mask = warp(mask, target, pose_coordinates, near, img.shape)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    

    im_floodfill = mask.copy()
    h, w = im_floodfill.shape[:2]
    mask1 = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask1, (mask.shape[1]-10, mask.shape[0]-10), 255); 
    # Invert floodfilled image
    mask = 255-cv2.bitwise_not(im_floodfill)
    
    #for point in target: cv2.circle(img, (point[0],point[1]), 15, (255,255,255), 2)
    #for point in pose_coordinates: cv2.circle(img, (point[0],point[1]), 30, (255,255,255), 3)
    
    #mask out source image
    img = cv2.bitwise_and(img, img, mask = mask)
    #add images
    comp = img + img2
    
    out.write(comp)
    #cv2.imshow('pnp', mask)
    cv2.imshow('comp', img+img2)
    
    k = cv2.waitKey(1)
    if k==27: break
    
cv2.destroyAllWindows()
cap.release()

