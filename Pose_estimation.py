# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing Libraries
import cv2
import mediapipe as mp
import time

'''
#######################
There are 33 landmarks
#######################

stati_image_mode----> False The solution treats the input images as a video stream
                      It will try to detect the most prominent person in the very first 
                      image and upon a successful detection further localizes the pos
                      landmarks. If set True person detection runs evey input image.
                      Default to False.

model_complexity----> Landmark accuacy as well as inference latency generally go up with the model complecity.
smooth_landmarks----> If true reduce jitter,but ignored if static_image_mode is also set to True

enable_segmentattion->  If True, the pose Landmark the solution also geenrates the segmentattion mask.
smooth Segmentation--> If set to True, reduce jitter

min_detection_condifence--> person-detection model for the detection to be considered successful. Default 0.5
min_traacking_confidence--> Landmarks-tracking for the pose landmrks to be considered tracked successfully. Default=0.5 

'''
class poseDetector():
    def __init__(self,mode=False,complexity=1,landmark=True,esegmentation=False,ssegmentation=True,
                detectionCon=0.5,trackCon=0.5):
    
        self.mode=mode
        self.complexity=complexity
        self.landmark=landmark
        self.esegmentation=esegmentation
        self.ssegmentation=ssegmentation
        self.detectionCon=detectionCon
        self.trackCon=trackCon
       
        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.complexity,self.landmark,self.esegmentation,self.ssegmentation,
                                   self.detectionCon,self.trackCon)
    
    # Function to find Pose
    def findPose(self,img,draw=True):
        
        # Convert image from BGR to RGB
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
    
        #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            
            # If draw ==True then draw.
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
       
        return img
    
    # Function to find the position of landmarks
    def findPosition(self,img,draw=True):
        
        lmList=[]
        if self.results.pose_landmarks:
            
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                
                # If draw==True them draw circles.
                if draw:
                    cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
        return lmList
            
    
def main():
    # Use webcam
    cap=cv2.VideoCapture(0)
    
    pTime=0
    detector=poseDetector()
    
    while True:
        # Read image from video
        success,img=cap.read()
        img=detector.findPose(img)
        lmList=detector.findPosition(img)
        # Printing the landmark number 14. Right elbow
        print(lmList[14])
        
        # Draw big circle on right elbow.
        cv2.circle(img,(lmList[14][1],lmList[14][2]),5,(255,0,0),cv2.FILLED)
       
        # Calculate the FPS of video.
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        
        # Write the FPS on screen
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,4,(255,0,0),4)
        
        # Show the results.
        cv2.imshow("Image",img)
        cv2.waitKey(1)
        
if __name__=="__main__":
    main()