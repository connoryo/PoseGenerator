import cv2
import mediapipe as mp
import numpy as np
from numpy import savetxt
import pandas as pd


class poseDetector():

    def __init__(self, mode=False, modelComp = 2, smooth=True, enable_seg=False,
                 smooth_seg=False, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.modelComp = modelComp
        self.smooth = smooth
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComp, self.smooth, self.enable_seg, self.smooth_seg,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:               
                IsPose = True
        else:
             IsPose = False
        return img, IsPose

    def findPosition(self, img, draw=True):
        self.lmList = []
        self.world = self.results.pose_world_landmarks.landmark
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    if id != 0 and id != 1 and id != 2 and id != 3 and id != 4 and id != 5 and id != 6 and id != 7 and id != 8 and id != 9 and id != 10 and id != 17 and id != 18 and id != 19 and id != 20 and id != 21 and id != 22:
                        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                        cv2.circle(img, (cx, cy), 15, (0, 0, 255), 2)

            cv2.line(img,self.lmList[16][1:],self.lmList[14][1:],(255,255,255),2)
            cv2.line(img,self.lmList[12][1:],self.lmList[14][1:],(255,255,255),2)
            cv2.line(img,self.lmList[12][1:],self.lmList[14][1:],(255,255,255),2)
            cv2.line(img,self.lmList[12][1:],self.lmList[24][1:],(255,255,255),2)
            cv2.line(img,self.lmList[12][1:],self.lmList[11][1:],(255,255,255),2)
            cv2.line(img,self.lmList[24][1:],self.lmList[26][1:],(255,255,255),2)
            cv2.line(img,self.lmList[24][1:],self.lmList[23][1:],(255,255,255),2)
            cv2.line(img,self.lmList[26][1:],self.lmList[28][1:],(255,255,255),2)
            cv2.line(img,self.lmList[28][1:],self.lmList[30][1:],(255,255,255),2)
            cv2.line(img,self.lmList[28][1:],self.lmList[32][1:],(255,255,255),2)
            cv2.line(img,self.lmList[32][1:],self.lmList[30][1:],(255,255,255),2)
            cv2.line(img,self.lmList[15][1:],self.lmList[13][1:],(255,255,255),2)
            cv2.line(img,self.lmList[13][1:],self.lmList[11][1:],(255,255,255),2)
            cv2.line(img,self.lmList[11][1:],self.lmList[23][1:],(255,255,255),2)
            cv2.line(img,self.lmList[23][1:],self.lmList[25][1:],(255,255,255),2)
            cv2.line(img,self.lmList[25][1:],self.lmList[27][1:],(255,255,255),2)
            cv2.line(img,self.lmList[27][1:],self.lmList[31][1:],(255,255,255),2)
            cv2.line(img,self.lmList[27][1:],self.lmList[29][1:],(255,255,255),2)
            cv2.line(img,self.lmList[31][1:],self.lmList[29][1:],(255,255,255),2)

        return self.lmList, self.world


# Video
cap = cv2.VideoCapture('Video/squat.mp4') # Change file path accordingly
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
Frame_rate = 30.0 # How fast video plays
File_name = 'video.avi' # Video file name
out = cv2.VideoWriter(File_name, cv2.VideoWriter_fourcc(*'XVID'), Frame_rate, (frame_width, frame_height))


detector = poseDetector()
Data3D = []
xLabel = []
yLabel = []
zLabel = []

for i in range(33):
    xLabel.append(str(i)+"X")
    yLabel.append(str(i)+"Y")
    zLabel.append(str(i)+"Z")

label3D = np.reshape(np.array([xLabel, yLabel, zLabel]),99,order='F')

while cap.isOpened():
    success, img = cap.read()
    if success:
        img, IsPose = detector.findPose(img, draw=True)

        if IsPose:
            lmList, world = detector.findPosition(img, draw=True)
            xData = []
            yData = []
            zData = []
            for i in range(33):
                xData.append(world[i].x)
                yData.append(world[i].y)
                zData.append(world[i].z)

            npArrF = np.reshape(np.array([np.array(xData),np.array(yData),np.array(zData)]),99,order='F')

            Data3D.append(npArrF)

        else:
            continue

        out.write(img)

        pd.DataFrame(Data3D,columns=label3D).to_csv('Data3D.csv')

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    else:
        break

cap.release()
out.release()

