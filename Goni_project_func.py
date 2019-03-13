#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 20:33:52 2019

@author: gonina
"""
#imports:
import cv2   ### must use openCV 4.0.0
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import statistics
from skimage import morphology


   ### background func ###
   
# define background image (to seprate image from background later)
# bg_image can be a given image or None
# VideoFile can be the video from which first frame will be used as background image or None
def background(bg_image, videoFile):
    
    if bg_image == None:  #if no background image had been given- use first frame as background image
        video = cv2.VideoCapture(videoFile) #import video
        for i in range(0, 1) :  #read first frame
            check, frame = video.read()  #read frame
            bg = frame
            # convert from bgr to gray
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            #binary threshold
            th2 = cv2.adaptiveThreshold(bg,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2) 

            #morphological 
            bg = cv2.erode(th2,morphology.disk(1),iterations = 1)
            bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, morphology.disk(1))
            bg = cv2.dilate(bg, morphology.disk(4), iterations = 1)
            bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, morphology.disk(5))
        
            #morphological #2
            bg2 = cv2.erode(bg,morphology.disk(1),iterations = 1)
            bg2 = cv2.morphologyEx(bg2, cv2.MORPH_OPEN, morphology.disk(1))
            bg2 = cv2.dilate(bg2, morphology.disk(4), iterations = 1)
            bg2 = cv2.morphologyEx(bg2, cv2.MORPH_OPEN, morphology.disk(10))
            bg2 = cv2.dilate(bg2, morphology.disk(1), iterations = 1)
        
        bg = bg+bg2
            
    else:   # if background image had been given- use it as background image
        bg = cv2.imread(bg_image) 
        # convert from bgr to gray
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        #binary threshold
        th2 = cv2.adaptiveThreshold(bg,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2) 
        #morphological 
        bg = cv2.erode(th2,morphology.disk(1),iterations = 1)
        bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, morphology.disk(1))
        bg = cv2.dilate(bg, morphology.disk(4), iterations = 1)
        bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, morphology.disk(5))
        
            #morphological #2
        bg2 = cv2.erode(bg,morphology.disk(1),iterations = 1)
        bg2 = cv2.morphologyEx(bg2, cv2.MORPH_OPEN, morphology.disk(1))
        bg2 = cv2.dilate(bg2, morphology.disk(4), iterations = 1)
        bg2 = cv2.morphologyEx(bg2, cv2.MORPH_OPEN, morphology.disk(10))
        bg2 = cv2.dilate(bg2, morphology.disk(1), iterations = 1)
        
        bg = bg+bg2
        bg = bg>2 #mask
    
    return bg
        

   ### timeStamp func ###

def timeStamp (frame, count, fps):
# draws timestamp on the frame
    time = count/fps
    cv2.putText(frame, "{}".format(time), (360, 500),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return time




  ### watershed version1 ###
# func that does watershed
# get frame = frame, bg_img = output of bg func
def watershed (frame, bg_img):
    # convert frame from rgb to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # binary threshold
    th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2) 
    # remove background noise
    th = th2-(bg_img)
    # #morphological #1  #remove noise
    clean = cv2.erode(th,morphology.disk(1),iterations = 1)  #erode
    clean2 = cv2.morphologyEx(clean, cv2.MORPH_OPEN, morphology.disk(1))  #opening
    clean3 = cv2.dilate(clean2, morphology.disk(4), iterations = 1)  #dilate
    clean4 = cv2.morphologyEx(clean3, cv2.MORPH_OPEN, morphology.disk(7)) #opening
    clean5 = cv2.erode(clean4,morphology.disk(2),iterations = 3)  #erode
    
    sure_fg = clean5>2  #mask
    # noise removal
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(clean4,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=1)
    sure_bg = sure_bg>2  # mask
    sure_bg = np.uint8(sure_bg)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # initialize empty image
    img = np.zeros([540,720,3], dtype='uint8') 
    img[:,:,:] = np.ndarray.astype(norm_data(frame)*255,'uint8')
    
    # check connected components and label
    ret, markers = cv2.connectedComponents(sure_fg)
    del ret
    # make backgorund 1, and unknown 0
    markers += 1
    markers[unknown==1] = 0
    # run watershed
    markers_after_WS = cv2.watershed(img,markers)
    
    return markers_after_WS
    


   ### find_contours ###
## contours detection and drawing
def find_contours (frame, markers): # markers = markers from watershed func
    centers=[] # empty list for bats center points locations
    num_items = np.max(markers)  
    # markeres start at 2 so actual number of items is num_items-1
    # print ((num_items-1), "items were detected")
    if num_items > 1:  # if found any items
        for i in range(2, num_items+1):  # markeres start at 2
            item = np.uint8(markers==i)
            # find contours:
            (major, minor, _) = cv2.__version__.split(".")
            if major == '4' or major == '2' :
                contours,_ = cv2.findContours(item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for cnt in contours:        
                    (x,y),radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x),int(y))
                    radius = int(radius)
                    if radius>11 and radius<60:
                        cntX = center[0]
                        cntY = center[1]
                        centers.append([cntX,cntY])  # append location of the bat to the list
                        # drawing a red circle around the bat:
                        frame = cv2.circle(frame, center,radius,(0,0,255),2)
                        
            else:
                   _,contours,_ = cv2.findContours(item, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                   for cnt in contours:        
                       (x,y),radius = cv2.minEnclosingCircle(cnt)
                       center = (int(x),int(y))
                       radius = int(radius)
                       if radius>11 and radius<60:
                           cntX = center[0]
                           cntY = center[1]
                           centers.append([cntX,cntY])  # append location of the bat to the list
                           # drawing a red circle around the bat:
                           frame = cv2.circle(frame, center,radius,(0,0,255),2)
                    
    return frame, centers



   ### segmention function ###
   # not in use in the final project

def segment(frame ,bg):
    # import image(frame)
    #img_bgr = cv2.imread(frame)
    img_bgr = frame
    # convert frame from rgb to gray
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    #binary threshold
    th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2) 

    #morphological #1
    mask1 = cv2.erode(th2,morphology.disk(1),iterations = 1)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, morphology.disk(1))
    mask1 = cv2.dilate(mask1, morphology.disk(4), iterations = 1)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, morphology.disk(5))

    # remove noise from background
    mask2=mask1-bg  #remove background

    #morphological #2
    mask2 = cv2.erode(mask2,morphology.disk(1),iterations = 1)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, morphology.disk(1))
    mask2 = cv2.dilate(mask2, morphology.disk(4), iterations = 1)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, morphology.disk(10))
    mask2 = cv2.dilate(mask2, morphology.disk(1), iterations = 1)

    mask3 = mask2 >=250  #creates a mask
    mask3 = mask2*mask3  #applies the mask to image

    #### contours detection and drawing
    centers=[]
    _,contours,_ = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:        
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
#        if radius>10 and radius<30:
        if radius>11 and radius<60:
#        if radius>15 and radius<60:
        # Find center point of contours:
            cntX = center[0]
            cntY = center[1]
            centers.append([cntX,cntY])
        
         # adding bats numbers in the center of each bat in video
#            numBats = len(centers)
#            for i in M:
#             cv2.putText(img_bgr, "{}".format(numBats), (cntX - 20, cntY - 20),
#                cv2.putText(img_bgr, "{}".format(numBats), (cntX, cntY),
#                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 
            img_bgr = cv2.circle(img_bgr,center,radius,(0,0,255),2)

    return img_bgr, centers



  ### DistanceCulc ###

def DistanceCalc (cntx_n1, cnty_n1, cntx_n2, cnty_n2):
    dx = cntx_n2 - cntx_n1
    dy = cnty_n2 - cnty_n1
    Distance = np.sqrt(dx*dx+dy*dy)

    return Distance

   ### DistanceCalcAllBats ###
# calculating distance to each bat in the following frame
#(BatsInFrames = dict, frame_num = keys, bat = bat index ,bat_dict1 = dict the func write into)
def DistanceCalcAllBats (BatsInFrames, frame_num, bat, bat_dict1):  
    bat_i_n = ["bat%02d" %i for i in range(1,21)]
    
    for bat in bat_i_n:
        bat_dict1['frame'+str(frame_num+1)][bat] = {}
        
        if bool (BatsInFrames.get(frame_num).get(bat)):# if there are bats in frame

            cntx_n1 = BatsInFrames.get(frame_num).get(bat)[0]  #value x of center point of a bat
            cnty_n1 = BatsInFrames.get(frame_num).get(bat)[1]  #value y of center point of a bat
    
            for bat_n in bat_i_n:
                try:  # value x,y of center point of a bat in the next frame
                     cntx_n2, cnty_n2 = BatsInFrames[frame_num+1][bat_n] 
                    # the dictionary is empty
                except KeyError:
                    continue 

                dx = cntx_n2 - cntx_n1
                dy = cnty_n2 - cnty_n1
                Distance = np.sqrt(dx*dx+dy*dy)
        
                # create all distance dict:
                bat_dict1['frame'+str(frame_num+1)][bat]['d_'+str(bat)+'_'+str(bat_n)] = Distance 
                
                # print (bat)
                # print (bat_dict1)
            
    return bat_dict1


        ### FindDistance ###
        # the distance that each bat did in following frames saved into bats dict
        #bat_dict = bats distances between frames dict batsM_dict = empty dict
def FindDistance(bat_dict, batsM_dict): 
#    batsM1 = defaultdict(dict)
    th = 4 # distance threshold (in pixels)
    bat_i = ["bat%02d" %i for i in range(1,21)] # indexing bats (for 20 bats)
    bat_dict_k = list(bat_dict.keys())  #list of all bat_dict keys (frames)
    for k in bat_dict_k:  # goes through all the frames in bat_dict
        for bat in bat_i: # for each bat distances list
            try:
                minkey = min(bat_dict[k][bat], key=bat_dict[k][bat].get) # finds the min distance
            except ValueError:
                continue
                 
            if bat_dict[k][bat][minkey] <= th: 
                batsM_dict[bat][k] = bat_dict[k][bat][minkey]  # save to dict
    return batsM_dict


   ### calcDistance  ###
# calculate the distance that each bat did during the night
   # batsM is batsM_dict from previous func, distance_result is the dict that func write into
def calcDistance (batsM, distance_result):
    batsD_k = list(batsM.keys())  #list of all batsM keys (bats)
    for bat_k in batsD_k:
        try:
            distance = sum(batsM[bat_k].values())
        except ValueError:
                continue
        distance_result[bat_k] = distance
    
    return distance_result


  ### calcMovement ###
  # calc movement for each bat (without frame names)
  # movement_results is the dict tje func write into
def calcMovement (batsM, movement_results):
    batsM_k = list(batsM.keys())  #list of all batsM keys (bats)
    for bat_k in batsM_k:
        try:
            movement = batsM[bat_k].values()
        except ValueError:
                continue
        movement_results[bat_k] = movement
    
    return movement_results


  ### plotDistance ###
# func that plot the total distance during the night of all the bats
  
def plotDistance (results_D):
    fig_D, ax = plt.subplots()
    plt.bar(results_D.keys(),height = results_D.values(),width=0.8)
    plt.title ('Bats total distance during the night')
    plt.xlabel ('Bats')
    plt.ylabel ('Distance (in pixels)')
    plt.tight_layout()
    plt.savefig('bats_total_D.png')
#    plt.show()
    return fig_D


   ### calcMeanMovement ###
#calaulate the mean movement
#     mean_movement is the dict tje func write into
def calcMeanMovement (batsM, mean_movement):  
    batsM_k = list(batsM.keys())  #list of all batsM keys (bats)
    for bat_k in batsM_k:
        try:
            movement = statistics.mean(batsM[bat_k].values())
        except ValueError:
                continue
        mean_movement[bat_k] = movement
    
    return mean_movement


   ### plotMeanMovement ###
# plots the mean movement of each bat during the night
   
def plotMeanMovement (mean_movement):
    fig_M, ax = plt.subplots()
    x = mean_movement.keys()
    y = mean_movement.values()
    plt.bar(x,y,width=0.8)
    plt.title ('Bats mean movment')
    plt.xlabel ('Bats')
    plt.ylabel ('mean movment (in pixels)')
    plt.savefig('mean_movement.png')
    plt.tight_layout()
#    plt.show()
    return fig_M


   ### plotMovement ###
# plot one bat level of movement- scatter plot
   
def plotMovement (batsM , bat):   
    keys = batsM[bat]
    N = len(keys)
    
    fig_movement = plt.figure()
    plt.scatter(batsM[bat].keys(), batsM[bat].values())
    
    plt.title('{} Level of movement during the night'.format(bat))
    plt.xlabel('Frames')
    plt.ylabel('Level of movement (in pixels)')
    plt.xticks(rotation='vertical')
    plt.ylim(0,5)

    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2 # inch margin
    s = maxsize/plt.gcf().dpi*N+2*m
    margin = m/plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    plt.tight_layout()
    plt.savefig('{} movementLevel.png'.format(bat))
#    plt.show()





   ### functions that aren't in use in the final project:  ###
   
   
   ###  DistanceFromMedian func ###
   
# Find the distance (in pixels) D between the bats locations and median location:
def DistanceFromMedian(numBats, centers):
    centers = np.asarray(centers)  #convert list to np array
    medLoc = np.median(centers, axis = 0) #median location
    if numBats >=2:
        for i in range(numBats):
            dx = centers[i][0] - medLoc[0]
            dy = centers[i][1] - medLoc[1]
            Dmed = np.sqrt(dx*dx+dy*dy)
            # print(i+1 ,":" ,Dmed)
            
    return Dmed


   ###  DistanceFromMean func ###
   
# Find the distance (in pixels) D between the bats locations and mean location:
def DistanceFromMean(numBats, centers):
    centers = np.asarray(centers)  #convert list to np array
    meanLoc = np.mean(centers, axis = 0) #mean location
    if len(centers) >=2:
        for i in range(numBats):
            dx = centers[i][0] - meanLoc[0]
            dy = centers[i][1] - meanLoc[1]
            Dmean = np.sqrt(dx*dx+dy*dy)
            # print(i+1 ,":" ,Dmean)
            
    return Dmean


  ### norm_data ###
# normalize image data to be between 0 to 1
def norm_data (image):
    image = image - np.min(image)
    image = image/np.max(image)

    return image  