import cv2
import vtk
import numpy as np
import toolsInImageClass as toolsInImage
import PointPairsClass as ptsPairs
import poseEstimateClass as poseEstimate
import StereoCameraCalibration as StereoCamCalib
import visualizeTools3D as vis
import csv

calib = StereoCamCalib.StereoCalibration('C:/Users/gupta/OneDrive/Desktop/Ankit/Project/Endoscope Calibration/')

fundamentalMatrix = calib.camera_model['F']
leftProjectionMatrix = calib.stereo_model['P1']
rightProjectionMatrix = calib.stereo_model['P2']

cap = cv2.VideoCapture('greenBarsTest.avi')
frameCount = 0


videotoolTransforms = []
videoPtCloud = []
while (cap.isOpened()):
    frameCount = frameCount + 1  # TODO: counter to count frame value
    print("frameCount", frameCount)
    ret, frame = cap.read()  # TODO: change varible ret to isFramePresent

    if frame is None:
        print(frameCount)

    leftImage = frame[0:540, 480:1440, :]  # TODO: keep variable for dimension specification
    rightImage = frame[540:1080, 480:1440, :]  # TODO define variable for dimension specification

    leftTools = toolsInImage.InstrumentsInImage(leftImage, 2)
    leftImageTools = leftTools.getImageTools()

    rightTools = toolsInImage.InstrumentsInImage(rightImage, 2)
    rightImageTools = rightTools.getImageTools()

    correspondingPoints = ptsPairs.PointPairsClass(leftImageTools, rightImageTools, fundamentalMatrix)
    matchedLeftCurves = correspondingPoints.leftImgMatchedPoints
    matchedRightCurves = correspondingPoints.rightImgMatchedPoints


    for i in range(0, len(matchedLeftCurves)):
        for j in range(0, len(matchedLeftCurves[i])):
            for k in range(0, len(matchedLeftCurves[i][j])):
                cv2.circle(leftImage, (int(matchedLeftCurves[i][j][k, 0]), int(matchedLeftCurves[i][j][k, 1])), 3,
                           (0, 0, 255), -1)
    cv2.imshow('image', leftImage)

    poseEst = poseEstimate.poseEstimateClass(matchedLeftCurves, matchedRightCurves, leftProjectionMatrix, rightProjectionMatrix)
    poseEst.getImgToolPoses()
    # imgToolsObj = poseEst(leftImgMatchedPoints, rightImgMatchedPoints)
    imgToolPoses = poseEst.imgToolPose
    scenePtCloud = poseEst.imgPointCloud
    pt = np.concatenate((scenePtCloud[0], scenePtCloud[1]), axis=1)
    if np.shape(pt)[1] < 400:
        tempMat = np.zeros([3, 400 - np.shape(pt)[1]])
        pt = np.hstack((pt, tempMat))
    if np.shape(pt)[1] > 400:
        pt = pt[:, :400]

    videoPtCloud.append(pt)
    videotoolTransforms.append(imgToolPoses)
    # cylinder.setCylinder()
    # tool1.setTransform(imgToolPoses[0])
    # tool2.setTransform(imgToolPoses[1])
    # print("toolPose1", imgToolPoses[0])
    # print("toolPose1", imgToolPoses[1])

    # cb.transform1 = imgToolPoses[0]
    # cb.transform2 = imgToolPoses[1]


    # cb = vis.vtkTimerCallback()
    # cb.actor = pointCloud.vtkActor
    # renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
    # renderWindowInteractor.CreateRepeatingTimer(200)
    # renderWindowInteractor.CreateRepeatingTimer(200)

    # Begin Interaction

    # renderWindowInteractor.Start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

np.save('toolPtCloud.npy', videoPtCloud)
np.save('toolTransforms.npy', videotoolTransforms)


