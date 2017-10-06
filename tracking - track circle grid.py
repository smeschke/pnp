import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/home/stephen/Desktop/tracking/coords.MP4')
output_folder = '/home/stephen/Desktop/tracking/data/'

win_size = 20
lk_params = dict(winSize = (win_size, win_size),
                 maxLevel = 3,
                 criteria = (cv2.TERM_CRITERIA_EPS |
                             cv2.TERM_CRITERIA_COUNT, 5, 0.03))

import math
def distance(a,b): return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

#get initial positions
shape = (4,4)
frame_number = 0
positions_list = []
#main loop
while True:    
    #read frame of video
    try: pimg  = pristine.copy()
    except: pass
    _, img = cap.read()
    try: pristine = img.copy()
    except: break

    #if a grid is found, use the grid
    #if no grid is found, tracking will be used later
    found, centers = cv2.findCirclesGrid(img, shape)
    if found and frame_number<100: tracking_list = centers
                 
    #no grid was found, use tracking to find the positions instead
    else:
        #make a tracking list
        tracking_list = []

        #try/except to see if video is over, and convert frame to gray
        old_gray = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        #track each
        for p0 in positions_list[-1]:
            #try and track using opticalk flow
            p1, error, _ = cv2.calcOpticalFlowPyrLK(old_gray, img_gray,p0, None, **lk_params)

            #check against the found corners
            if found:
                for center in centers:
                    dist = distance(center[0], p1[0])
                    #if the optical flow tracking result is really near a found center,
                    #use the found center value instead of the optical flow tracking value
                    if dist<5:
                        #print 'point corrected',p1, center, dist
                        p1 = center
                        
                        
            #convert the coordinates to two intigers
            xy = int(p1[0][0]), int(p1[0][1])
            cv2.circle(img, xy, win_size, (255,0,0), 1)
            xy = int(p0[0][0]), int(p0[0][1])
            cv2.circle(img, xy, win_size, (0,0,0), 1)
            #new click, update p0
            tracking_list.append(p1)
        

    positions_list.append(tracking_list)
    
    for p1 in positions_list[-1]:
        xy = int(p1[0][0]), int(p1[0][1])
        cv2.circle(img, xy, 2, (0,0,255), 2)
    
    #show the frame and wait
    cv2.imshow('img', img)
    k = cv2.waitKey(1)
    #if keypress is "esc"
    if k == 27: break
    #increment frame number
    frame_number += 1

#close the window
cv2.destroyAllWindows()
print len(positions_list), frame_number, len(positions_list[0])
print 'finished tracking'
for idx in range(len(positions_list[0])):
    output_path = output_folder+str(idx)+'.csv'
    #write data
    import csv
    with open(output_path, 'w') as csvfile:
        fieldnames = ['x_position', 'y_position']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        for positions in positions_list:
            position = positions[idx]
            try: x, y = position[0][0], position[0][1]
            except: x, y = positions[0][0][0], position[0][0][1]
            writer.writerow({'x_position': x, 'y_position': y})

print 'finished writing data'
