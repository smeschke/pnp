import cv2
import numpy as np
import math
import os
import pandas as pd
from scipy import signal
#signal.savgol_filter(x, 25, 3)
#captrue webcam
pose_folder = '/home/stephen/Desktop/pnp files/tablefront/'
source_folder = '/home/stephen/Desktop/pnp files/red/'
cap = cv2.VideoCapture(pose_folder + 'pose.avi')
fps = 60.0
cols, rows = 1280,720
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/stephen/Desktop/pnp_out.avi', fourcc, fps, (cols, rows))

num_images  = len(os.listdir(source_folder+'images/'))

#the tracking data is not perfect, so it needs to be smoothed
#the higher the number the more smoothing (range is >5, 25 is good
fv = 13

#get data for how the object is posed
files = ['six.csv','three.csv','twelve.csv','nine.csv']
data = []
for f in files:
    df = pd.read_csv(source_folder + f)
    a,b = zip(*df.values)    
    vals = zip(signal.savgol_filter(a, fv, 3), 500+signal.savgol_filter(b, fv,3))#add 500 here for anamorphic poses
    data.append(vals)
source_data = zip(data[0],data[1],data[2],data[3])
#get data for the location of the white board in the background image
files = ['posesix.csv','posethree.csv','posetwelve.csv','posenine.csv']
data = []
for f in files:
    df = pd.read_csv(pose_folder + f)
    print len(df.values), ' values in ', f
    a,b = zip(*df.values)
    vals = zip(signal.savgol_filter(a, fv, 3), signal.savgol_filter(b, fv,3))
    data.append(vals)
pose_data = zip(data[0],data[1],data[2],data[3])

def get_slope(a,b):
    try: return (float(a[1])-b[1])/(a[0]-b[0])
    except: return 0

def resize(target, source, source_img):
    for i in source:
        xy = (int(i[0]),int(i[1]))
        #print xy
        #cv2.circle(source_img, xy, 12, (123,255,123), 2)
    target_center, target_size = cv2.minEnclosingCircle(np.array(target, np.float32))
    source_center, source_size = cv2.minEnclosingCircle(np.array(source, np.float32))
    #print source, target_size, source_size
    #source = source*target_size/source_size
    bg = np.zeros((800,1000,3), np.uint8)
    #print target_size, source_size
    sss=source_size/target_size
    for i in target:
        xy = (int(i[0]),int(i[1]))
        #print xy
        #cv2.circle(bg, xy, 2, (123,123,123), 2)
    resize_coords = []
    for i in source:
        xy = (int(i[0]/sss),int(i[1]/sss))
        resize_coords.append(xy)
        #print xy
        #cv2.circle(bg, xy, 2, (123,255,123), 2)
    h,w,_ = source_img.shape
    source_img = cv2.resize(source_img, (int(w/sss), int(h/sss)))
    #cv2.imshow('source_img', source_img)
    #print source_img.shape
    #cv2.imshow('bg',bg)
    #cv2.waitKey(0)
    return source_img, resize_coords

def center_image(target, source, background, foreground):
    target_center, target_size = cv2.minEnclosingCircle(np.array(target, np.float32))
    source_center, source_size = cv2.minEnclosingCircle(np.array(source, np.float32))
    dx = int(target_center[0] - source_center[0])
    dy = int(target_center[1] - source_center[1])
    fg_h, fg_w,_ = foreground.shape
    x_buffer = int(fg_w - source_center[0])
    y_buffer = int(fg_h - source_center[1])
    #print x_buffer, y_buffer, dx, dy, dy-y_buffer,'to',dy-y_buffer+fg_h, dx-x_buffer,'to',dx-x_buffer+fg_w, foreground.shape, background.shape
    bg = np.zeros_like(background)
    cv2.imshow('f',foreground)

    if dx<0:
        foreground = foreground[0:, dx:]
        dx = 0
    if dy<0:
        foreground = foreground[dy:, 0:]
        dy = 0
    if dx+fg_w>1280:
        foreground = foreground[0:, 0:fg_w-(1280-(dx+fg_w))]
    if dy+fg_h>720:
        foreground = foreground[0:fg_h-(720-(dy+fg_h))]
    h,w,_ = foreground.shape
    bg[dy:dy+h, dx:dx+w] = foreground
    #cv2.imshow('bg', bg)
    #cv2.waitKey(0)
    return bg

def fill(mask):
    im_floodfill = mask.copy()
    h, w = im_floodfill.shape[:2]
    mask1 = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask1, (1080,719), 255); 
    # Invert floodfilled image
    mask = cv2.bitwise_not(im_floodfill)
    #cv2.imshow('mask', mask)
    #cv2.waitKey(0)
    return mask
    
frame_number = 0
while True:
    #grab image
    _, img = cap.read()

    #get target in source (white board)
    target = pose_data[frame_number]

    #calculate rotation of the white board
    a,b = target[0], target[2]
    t =  math.atan(-get_slope(a,b))
    if a[0]>b[0]: rotation = .75-t/(2*math.pi)
    else: rotation = .25-t/(2*math.pi)
    #print 'rotation', rotation

    #find the image that most closely matches the amount of rotation
    near = int(num_images*rotation)

    #get coordinates for the image that most closely matches the amount of rotation
    #the pose of the object
    pose_coordinates = source_data[int(len(source_data)*rotation)]

    #nearest matching image
    pnp = cv2.imread(source_folder+'images/'+str(near)+'.png')
    mask = cv2.imread(source_folder+'masks/'+str(near)+'.png')
    
    
    
    #resize the pose_coordinates so they they match the size ofthe object
    source_img, resize_coords = resize(target, pose_coordinates, pnp)
    foreground = center_image(target, resize_coords, img, source_img)

    #do the same for the mask
    mask, _  = resize(target, pose_coordinates, mask)
    mask = center_image(target, resize_coords, img, mask)
    #_, thresh = cv2.threshold(mask, 123,2255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    mask = fill(mask)
    _, mask = cv2.threshold(mask, 123, 255, cv2.THRESH_BINARY)
    #print mask.shape
    #create composite image
    img = cv2.bitwise_and(img, img, mask = 255-mask)
    img += foreground

    
    #cv2.imshow('mask', mask)
    cv2.imshow('comp', img)
    out.write(img)
    frame_number+=1
    k = cv2.waitKey(1)
    if k==27: break
    
cv2.destroyAllWindows()
cap.release()
