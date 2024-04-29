import os
import tensorflow
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

from time import time
import socket
from goprocam import GoProCamera, constants

WRITE = False
gpCam = GoProCamera.GoPro()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t=time()
gpCam.livestream("start")
gpCam.video_settings(res='1080p', fps='30')
gpCam.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720)
cap = cv2.VideoCapture("udp://10.5.5.9:8554", cv2.CAP_FFMPEG)
counter = 0


classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)
classIDBin = 0
# Import all the waste images
imgWasteList = []
pathFolderWaste = "Resources/Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

# Import all the waste images
imgBinsList = []
pathFolderBins = "Resources/Bins"
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

# 0 = Recyclabler
# 1 = Hazardous
# 2 = Food
# 3 = Residual

classDic = {0: None,
            1: 1,
            2: 1,
            3: 2,
            4: 2,
            5: 1,
            6: 1,
            7: 3,
            8: 3}

while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))



    imgBackground = cv2.imread('Resources/background.png')

    predection = classifier.getPrediction(img)

    classID = predection[1]
    print(classID)
    if classID != 0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))

        classIDBin = classDic[classID]

    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))

    imgBackground[148:148 + 340, 159:159 + 454] = imgResize
    # Displays
    # cv2.imshow("Image", img)

    if WRITE == True:
        cv2.imwrite(str(counter) + ".jpg", img)
        counter += 1


    if time() - t >= 2.5:
        sock.sendto("_GPHD_:0:0:2:0.000000\n".encode(), ("10.5.5.9", 8554))
        t = time()
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)