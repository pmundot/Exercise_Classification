import pandas as pd
import pickle
import numpy as np
import math
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split

import mediapipe as mp
import cv2

class photo_classification():
    def __init__(self,img):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.7)
        self.img = img
        self.trans = {0:'Jumping Jacks',1:'Push Ups',2:'Squats'}
        self.exercise = {0:'Jumping Jacks',1:"Push-ups",2:"Body Weight Squats"}
        re = [12,14,16]
        rs = [14,12,24]
        rh = [12,24,26]
        rk = [24,26,28]
        ra = [26,28,32]
        self.right_side = [re,rs,rh,rk,ra]
        #Column names for future dataframes

        self.name = ['Exercise','right_elbow','right_shoulder','right_hip','right_knee','right_ankle']
    
    def get_img(self):
        self.image = cv2.imread(self.img)
        self.RGB = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(self.RGB)

    def get_landmark(self):
        self.get_img()
        self.ids = []
        self.x = []
        self.y = []
        self.z = []
        for i,poses in enumerate(self.results.pose_landmarks.landmark):
            h,w,d = self.image.shape
            self.x.append(int(poses.x*w))
            self.y.append(int(poses.y*h))
            self.z.append(int(poses.z*d))
            self.ids.append(i)
    
    def normal_image(self):
        self.get_img()
        cv2.imshow('Image',self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    
    def skeleton_image(self):
        self.get_img()
        self.mp_drawing.draw_landmarks(self.image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Image_Skel',self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    
    def angels_image(self):
        self.get_landmark()
        for i in self.right_side:
            self.angles_to_image(i)
        cv2.imshow('Image_ang',self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

    def angles_to_df(self,lst):
        self.get_landmark()
        one = lst[0] 
        two = lst[1]
        three = lst[2]
        once = self.angle_2p_3d((self.x[one],self.y[one],self.z[one]),(self.x[two],self.y[two],self.z[two]),(self.x[three],self.y[three],self.z[three]))
        return(once)

    def angles_to_image(self,lst):
        one = lst[0] 
        two = lst[1]
        three = lst[2]
        cv2.circle(self.image,(self.x[one],self.y[one]),4,(255,0,0),cv2.FILLED)
        cv2.circle(self.image,(self.x[two],self.y[two]),4,(255,0,0),cv2.FILLED)
        cv2.circle(self.image,(self.x[three],self.y[three]),4,(255,0,0),cv2.FILLED)
        once = self.angle_2p_3d((self.x[one],self.y[one],self.z[one]),(self.x[two],self.y[two],self.z[two]),(self.x[three],self.y[three],self.z[three]))
        cv2.putText(img=self.image, 
                    text=str(once), 
                    org=(self.x[two], self.y[two]), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=.5, color=(0, 255, 0),thickness=1)
    
    def angle_2p_3d(self,a, b, c):       
        v1 = np.array([ a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        v2 = np.array([ c[0] - b[0], c[1] - b[1], c[2] - b[2]])

        v1mag = np.sqrt([ v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]])
        v1norm = np.array([ v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag])

        v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
        v2norm = np.array([ v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag])
        res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
        angle_rad = np.arccos(res)

        return round(math.degrees(angle_rad),2)
    
    def save_img(self,name):
        cv2.imwrite('{}\\{}.png'.format(os.getcwd(),name),self.image)
    
    def class_image(self,tp,name):
        filename = 'finalized_model.sav'
        clf = pickle.load(open(filename, 'rb'))
        finall = np.reshape([self.angles_to_df(t) for t in self.right_side],(1,5))

        cv2.putText(img=self.image, 
                text=self.trans[int(clf.predict(finall))], 
                org=(70,70), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1.5, color=(0, 0, 0),thickness=2)
        
        if tp.strip().lower()[0]=='d':
            pass
        elif tp.strip().lower()[0]=='s':
            self.mp_drawing.draw_landmarks(self.image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        elif tp.strip().lower()[0]=='a':
            for i in self.right_side:
                self.angles_to_image(i)
        
        elif tp.strip().lower()[0]=='c':
            self.mp_drawing.draw_landmarks(self.image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            for i in self.right_side:
                self.angles_to_image(i)
        else:
            pass

        if name is not None:
            self.save_img(name)

        cv2.imshow('clap',self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class video_classification():
    def __init__(self,vid):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.7)
        self.vid = vid
        self.trans = {0:'Jumping Jacks',1:'Push Ups',2:'Squats'}

        re = [12,14,16]
        rs = [14,12,24]
        rh = [12,24,26]
        rk = [24,26,28]
        ra = [26,28,32]
        self.right_side = [re,rs,rh,rk,ra]
        #Column names for future dataframes

        self.name = ['Exercise','right_elbow','right_shoulder','right_hip','right_knee','right_ankle']
    
    def get_vid(self):
        self.ret, self.frame = self.vids.read()
        self.RGB = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(self.RGB)

    def get_landmark(self):
        self.ids = []
        self.x = []
        self.y = []
        self.z = []
        for i,poses in enumerate(self.results.pose_landmarks.landmark):
            h,w,d = self.frame.shape
            self.x.append(int(poses.x*w))
            self.y.append(int(poses.y*h))
            self.z.append(int(poses.z*d))
            self.ids.append(i)

    def angles_to_df(self,lst):
        one = lst[0] 
        two = lst[1]
        three = lst[2]
        once = self.angle_2p_3d((self.x[one],self.y[one],self.z[one]),(self.x[two],self.y[two],self.z[two]),(self.x[three],self.y[three],self.z[three]))
        return(once)

    def angles_to_image(self,lst):
        one = lst[0] 
        two = lst[1]
        three = lst[2]
        cv2.circle(self.frame,(self.x[one],self.y[one]),4,(255,0,0),cv2.FILLED)
        cv2.circle(self.frame,(self.x[two],self.y[two]),4,(255,0,0),cv2.FILLED)
        cv2.circle(self.frame,(self.x[three],self.y[three]),4,(255,0,0),cv2.FILLED)
        once = self.angle_2p_3d((self.x[one],self.y[one],self.z[one]),(self.x[two],self.y[two],self.z[two]),(self.x[three],self.y[three],self.z[three]))
        cv2.putText(img=self.frame, 
                    text=str(once), 
                    org=(self.x[two], self.y[two]), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=.5, color=(0, 255, 0),thickness=1)
    
    def angle_2p_3d(self,a, b, c):       
        v1 = np.array([ a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        v2 = np.array([ c[0] - b[0], c[1] - b[1], c[2] - b[2]])

        v1mag = np.sqrt([ v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]])
        v1norm = np.array([ v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag])

        v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
        v2norm = np.array([ v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag])
        res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
        angle_rad = np.arccos(res)

        return round(math.degrees(angle_rad),2)
    
    #def save_img(self,name):
    #    cv2.imwrite('{}\\{}.png'.format(os.getcwd(),name),self.image)
    
    def class_video(self,tp):
        filename = 'finalized_model.sav'
        clf = pickle.load(open(filename, 'rb'))

        self.vids = cv2.VideoCapture(self.vid)
        w = self.vids.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.vids.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.vids.get(cv2.CAP_PROP_FPS) 
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, fps, (int(w),int(h)))

        while True:
            self.get_vid()
            try:
                self.get_landmark()
                finall = np.reshape([self.angles_to_df(t) for t in self.right_side],(1,5))

                cv2.putText(img=self.frame, 
                    text=self.trans[int(clf.predict(finall))], 
                    org=(70,70), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1.5, color=(0, 0, 0),thickness=3)
                
                if tp.strip().lower()[0]=='d':
                    pass
                elif tp.strip().lower()[0]=='s':
                    self.mp_drawing.draw_landmarks(self.frame, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                elif tp.strip().lower()[0]=='a':
                    for i in self.right_side:
                        self.angles_to_image(i)
                
                elif tp.strip().lower()[0]=='c':
                    self.mp_drawing.draw_landmarks(self.frame, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    for i in self.right_side:
                        self.angles_to_image(i)
                else:
                    pass
            except AttributeError:
                pass

            #self.frame = cv2.resize(self.frame, (0, 0), fx = 0.3, fy = 0.3)
            cv2.imshow('frame', self.frame)
            out.write(self.frame) 
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self.vid.release()
        out.release()

        cv2.destroyAllWindows()   
