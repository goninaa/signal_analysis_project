#!/usr/bin/env python
# coding: utf-8

# In[ ]:
       ### must use openCV 4.0.0
  # Reads bats video, detects bats and circles them and save it into a new video.
  # finds bats locations in each frame. 
  # calculates each bat total distance during the night and plots all bats.
  # calculates each bat mean movement (between frames) and plots all bats
  # calculates each bat level of movement in each frame and plot it.
  

#imports:
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import statistics
from skimage import morphology

from Goni_project_func import (background, watershed, find_contours, segment, timeStamp, 
                             DistanceCalcAllBats, FindDistance, calcDistance, calcMovement, 
                             plotDistance, calcMeanMovement, plotMeanMovement, plotMovement, 
                             norm_data)
# In[1]:


# function that imports video, run the algorithms on each frame and display the results.
# Reads bats video, detects bats and circles them and save it into a new video.
# finds bats locations in each frame and save it into a dict.
#video is the video file, num_frame is the number of frame we want to read (0= all frames)
def readVideo (videoFile, num_frames):  
    
    # import video
    video = cv2.VideoCapture(videoFile)  #import video from file
    fps = video.get(cv2.CAP_PROP_FPS)  # get videos fps

    # Get the width and height of frame
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for mac
    out = cv2.VideoWriter('newBats.mp4', fourcc, fps, (width, height)) # for mac
#    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # for windows
#    out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height)) # for windows
    
    #define background
    bg = background(None, videoFile)  #defines the first frame as background
#     bg = background('frame0.jpg', None)  #defines image as background
    count = 0

    bats_loc = {}  # empty dict for locations of bats in frame
    BatsInFrames = {}  # empty dict for dict of all the loc of all the bats in all frames
    
    if num_frames == 0:
    # read the whole video
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  #number of frames in video
        for i in range(0, num_frames) :
            check, frame = video.read()  #read each frame
            # segmention and watershade for finding the bats:
            markers = watershed(frame, bg) 
            # contours detection, drawing and finding bats locations :
            frame, centers = find_contours (frame, markers) 
            #find and circle the bats in frame using erosion and dilation filters:
#            seg, centers = segment(frame, bg)  
            time = timeStamp (frame, count, fps) # draw timestamp on the frame
            
            # creates a dict of bats locations in each frame:
            bats_num = 20  #number of bats we want to save 
            bat_i = ["bat%02d" %i for i in range(1,bats_num+1)] # indexing bats (for 'bats_num' bats)
            bats_loc = dict(zip(bat_i, centers)) # locations of bats in frame
            BatsInFrames[i] = bats_loc # bats locations in each frame
#            print (BatsInFrames)
            count+=1  
            
            # write the frame into a new video file
#            out.write(frame)
            
            # display
#            cv2.imshow('bats detector', seg)
            # if pressed 'q' than close
#            cv2.waitKey(1)  
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        video.release()
        cv2.destroyAllWindows()
            

    else:
    # read a few frames
        for i in range(0, num_frames) :  #number of frames to read
            check, frame = video.read()  #read each frame
            # segmention and watershade for finding the bats:
            markers = watershed(frame, bg) 
            # contours detection, drawing and finding bats locations :
            frame, centers = find_contours (frame, markers) 
            #find and circle the bats in frame using erosion and dilation filters:
#            seg, centers = segment(frame, bg)  
            time = timeStamp (frame, count, fps) # draw timestamp on the frame
            
            # creates a dict of bats locations in each frame:
            bats_num = 20  #number of bats we want to save 
            bat_i = ["bat%02d" %i for i in range(1,bats_num+1)] # indexing bats (for 'bats_num' bats)
            bats_loc = dict(zip(bat_i, centers)) # locations of bats in frame
            BatsInFrames[i] = bats_loc # bats locations in each frame
#            print (BatsInFrames)
            count+=1
            
            # write the frame into a new video file
            out.write(frame)
                        
            # Display the resulting frame
#            cv2.imshow('bats detector', seg)
            # if pressed 'q' than close
#            cv2.waitKey(1)  #cheak    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        video.release()
        cv2.destroyAllWindows()

    
    return  BatsInFrames


# In[2]:

# reads video, detects bats, finding bats locations and save new video file:
BatsInFrames = readVideo('bats.mp4', 1500)  # recomended  1500-2000
 


# In[5]:


# calculating each bat distance to each bat in the following frame

bat_i = ["bat%02d" %i for i in range(1,51)] # indexing bats
frame_dict = {} # contain frames (to each bat- nested in bat_dict)
bat2batD = {}  # bat to bat distance (nested in frame_dict)
bat_dict1 = defaultdict(dict) #main dict
   
for frame_num, bat_loc in BatsInFrames.items():
    try:
        if BatsInFrames[frame_num] == {} or  BatsInFrames[frame_num+1] =={} :  
        # if no bats in that frame or the next one- continue to the next frame
            continue
    except KeyError:
        break
    else:   # if there are bats in this frame and in the next one
        for bat in bat_i:
            bat_dict = DistanceCalcAllBats (BatsInFrames,frame_num, bat, bat_dict1) #calculate the distance between bats


# In[8]:
                        
# creates a dict for each bat with the movement it did from frame to frame
batsM_dict = defaultdict(dict) # bats movement dict
batsM = FindDistance(bat_dict,batsM_dict)
# print (batsM)


# In[10]:

# calculate the distance that each bat did during the night
distance_result = defaultdict(dict) # empty dict
results_D = calcDistance(batsM, distance_result)
#print (results_D)


# In[12]:

#plot the total distance during the night of all the bats
plot = plotDistance (results_D)

# In[15]:

# calc and plot mean movement for all the bats
mean_movement = defaultdict(dict) #empty dict
mean_movement = calcMeanMovement (batsM, mean_movement)
plot2 = plotMeanMovement (mean_movement)


# In[19]:

# calc movement for each bat (for the movement plot next)
movement_results = defaultdict(dict) #empty dict
movement_results = calcMovement (batsM, movement_results)
print (movement_results)


# In[21]:

# plot each bat level of movement- scatter plot
bats_num = 20 # for how many bats to plot (plots only if they have data)
bats = ["bat%02d" %i for i in range(1,bats_num+1)]
for bat in bats:
    if batsM[bat] !={}:  # plot only for bats that have data
        plotMovement (batsM, bat)

plt.show()


