#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:28:02 2020

@author: guillermogarcia
"""


import cv2
import numpy as np
import os

dirs = os.listdir('shirts/shirt_dataset/unlabeled')
blue_shirts=[]
non_blue_shirts=[]
for i in dirs:

    img= cv2.imread("shirts/shirt_dataset/unlabeled/"+ str(i))   
    #Crop 20 % of the bottom part cause jeans bother
    x,y, _ = img.shape
    crop_img = img[0:int(y*0.9)]
    
    denoised = cv2.GaussianBlur(crop_img, (5, 5),cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    
    
    lower_blue = np.array([100, 30, 50]) 
    upper_blue = np.array([120, 255, 255]) 
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    #I take advantage of the characteristics of the problem .i.e All images have a
    #uniform blackground and shirts are properly framed. We can compare the number
    #of blue pixels in the image and conclude by a threshold whether the image 
    #contains a blue shirt color or not.
    
    counting = cv2.countNonZero(mask)
    dim_img = x*y
    ratio_blue_pixels = counting/dim_img
    print('ratio_blue_pixels: ',ratio_blue_pixels)
    
    if ratio_blue_pixels > 0.15:  
        blue_shirts.append(i)        
        print('imagen: ',i, 'is a blue shirt')
    else: non_blue_shirts.append(i)
    
    
    #Coment to not visualize images
    cv2.imshow('img', cv2.resize(crop_img, (960, 540)) )
    cv2.waitKey(0)
    cv2.imshow('res', cv2.resize(mask, (960, 540)) )
    cv2.waitKey(0)
    #------------------------
     
cv2.destroyAllWindows()

np.save('blue_shirts_detected.npy', blue_shirts)
np.save('non_blue_shirts_detected.npy', non_blue_shirts)


