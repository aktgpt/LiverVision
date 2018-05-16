import cv2
import numpy as np

import math
from scipy.optimize import minimize



class poseEstimateClass:

        def __init__(self, leftMatchedPoints, rightMatchedPoints, calibrationParameters):
            self.leftMatchedPoints = leftMatchedPoints
            self.rightMatchedPoints = rightMatchedPoints
            self.leftProjectionMatrix = calibrationParameters['LeftProjectionMatrix']
            self.rightProjectionMatrix = calibrationParameters['RightProjectionMatrix']
            self.leftCameraMatrix = calibrationParameters['LeftCameraMatrix']
            self.rightCameraMatrix = calibrationParameters['RightCameraMatrix']
            self.leftDistCoeffs = calibrationParameters['LeftDistortionCoefficients']
            self.rightDistCoeffs = calibrationParameters['RightDistortionCoefficients']
            self.rightCameraRotation = calibrationParameters['RightCameraRotation']
            self.rightCameraTranslation = calibrationParameters['RightCameraTranslation']
            self.imgToolPts = []
            self.toolPts = []
            self.toolLine = np.zeros([6, 1])
            self.initialDistLineFromOrigin = float(0)
            self.signOriginFromTool = np.sign(self.initialDistLineFromOrigin)
            self.toolPose = np.zeros([4, 4])
            self.imgPointCloud = []
            self.imgToolPose = []
            self.leftImageToolLines = []
            self.rightImageToolLines = []

        def get3DPts(self, leftPoints, rightPoints):
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_xlim(-50, 50)
            # ax.set_ylim(-20, 20)
            # ax.set_zlim(0, 250)
            scenePoints = []
            for i in range(0, len(leftPoints)):
                toolPoints3D = np.array([]).reshape(3, 0)
                meanToolCurves = []
                for j in range(0, len(leftPoints[i])):
                    if leftPoints[i][j] !=[]:
                        points3D = cv2.triangulatePoints(self.leftProjectionMatrix, self.rightProjectionMatrix,
                                                     np.transpose(leftPoints[i][j]), np.transpose(rightPoints[i][j]))
                        points3D = (points3D/points3D[3])[:3]
                        meanToolCurves.append(np.mean(points3D, axis=1))
                        toolPoints3D = np.hstack((toolPoints3D, points3D))
                    else:
                        continue
                # ax.scatter3D(toolPoints3D[0], toolPoints3D[1], toolPoints3D[2])
                # self.toolPts = self.removeOutliers(toolPoints3D, self.toolLine)
                meanToolCurves = np.transpose(np.asarray(meanToolCurves))
                self.imgToolPts.append(meanToolCurves)
                # self.setToolParameters(meanToolCurves)
                # self.updateToolLine()
                # self.setToolPose()
                # print(self.toolPose)
            # ax.scatter3D(toolCenterPoints[0], toolCenterPoints[1], toolCenterPoints[2])
                scenePoints.append(toolPoints3D)
            # plt.show()
            return scenePoints

        def getMeanPts(self, pts):
            meanMat = np.mean(pts, axis=1)
            return meanMat

        def setToolParameters(self, pts):
            self.initialDistLineFromOrigin = float(self.signedDistPtsFromLine(np.zeros([3, 1]), self.toolLine))
            self.signOriginFromTool = np.sign(self.initialDistLineFromOrigin)

        def setToolPose(self):
            pose = self.getPosefromLine(self.toolLine)
            self.toolPose = pose

        def updateToolLine(self):
            if self.toolPts !=[]:
                self.toolLine = self.optimizeLine(self.toolLine)
                self.toolLine[0:3] = np.mean(self.toolPts, axis=1)
            else:
                self.toolLine

        def signedDistPtsFromLine(self, pts, line):
            linePt = np.transpose(line[:3])
            if pts != []:
                ptsShape = np.shape(pts)
                if np.shape(ptsShape)[0] == 1:
                    numPts = 1
                else:
                    numPts = ptsShape[1]
                # distMat = np.empty([1, np.shape(pts)[1]])
                distMat = []
                dirVect = np.transpose(line[3:6])
                signMat = []
                for i in range(0, numPts):
                    #ptshape = np.shape(pts[:, i])
                    ptVect = pts[:, i] - linePt
                    crossProd = np.cross(ptVect, dirVect)
                    signMat.append(np.sign(np.sum(crossProd)))
                    distMat.append(np.linalg.norm(crossProd) / np.linalg.norm(dirVect))
                distMat = np.asarray(distMat)
                signMat = np.asarray(signMat)
                signedDistMat = distMat*signMat
            else:
                signedDistMat = float(0)
            return signedDistMat

        def errorFnLine(self, line):
            errorMat = np.abs(self.signedDistPtsFromLine(self.toolPts, line) + self.signOriginFromTool*2.5)
            errorFn = np.mean(errorMat)
            return errorFn

        def linePtCloud(self, pts3D):
            if pts3D !=[]:
                mean, eigenVectors = cv2.PCACompute(np.transpose(pts3D), mean=None)
                line = np.concatenate((np.ravel(mean), eigenVectors[0, :]), axis=0)
            else:
                line = np.array([0, 0, 0, 0, 0, 0])
            return line

        def constraintsLine(self):
            constraint = ({'type': 'eq',
                           'fun': lambda line: np.array([line[3]**2 + line[4]**2 + line[5]**2 - 1])})
                          # {'type': 'ineq',
                          #  'fun': lambda line: np.array([np.sqrt((line[5]*line[0]*line[1] - line[4]*line[0]*line[2])**2 +
                          #                                        (line[5]*line[0]*line[1] - line[3]*line[1]*line[2])**2 +
                          #                                        (line[4]*line[0]*line[2] - line[3]*line[1]*line[2])**2) -
                          #                                         self.initialDistLineFromOrigin-2])})
            return constraint



        def optimizeLine(self, line):
            initLine = line
            res = minimize(self.errorFnLine, initLine, constraints=self.constraintsLine(),
                           method='SLSQP')
            finalLine = res.x
            return finalLine

        def getPosefromLine(self, line):
            # phi, theta = self.getEulerAnglesFromDirection(line[3:6])
            # rotMat = self.rotationMatrixFromEulerAngles(phi, theta)
            rotMatRod = self.getRotationMatrixFromVector(line[3:6])
            dirVectX = line[3:6] / np.linalg.norm(line[3:6])
            dirVectY = np.array([0, 1, 0])
            dirVectZ = np.cross(dirVectX, dirVectY)
            dirVectYUpdated = np.cross(dirVectX, dirVectZ)
            transformMatrix = np.identity(4)
            for i in range(0, 3):
            #     transformMatrix[i, 0] = dirVectX[i] / np.linalg.norm(dirVectX)
            #     transformMatrix[i, 1] = dirVectYUpdated[i] / np.linalg.norm(dirVectYUpdated)
            #     transformMatrix[i, 2] = dirVectZ[i] / np.linalg.norm(dirVectZ)
                transformMatrix[i, 3] = line[i]
            transformMatrix[0:3, 0:3] = rotMatRod
            transformMatrix[3, 0:4] = [0, 0, 0, 1]
            # check = np.dot(transformMatrix[0, 0:3], transformMatrix[1, 0:3])
            # print("dot product of x and y: " + str(check))
            # check = np.dot(transformMatrix[1, 0:3], transformMatrix[2, 0:3])
            # print("dot product of x and z: " + str(check))
            # check = np.dot(transformMatrix[0, 0:3], transformMatrix[2, 0:3])
            # print("dot product of y and z: " + str(check))
            # print(np.linalg.norm(transformMatrix[0, 0:3]))
            # print(np.linalg.norm(transformMatrix[1, 0:3]))
            # print(np.linalg.norm(transformMatrix[2, 0:3]))
            # #self.toolPose = transformMatrix
            # fig = plt.figure(2)
            # ax = fig.add_subplot(111, projection='3d')
            #
            # ax.plot([line[0], line[0] + dirVectX[0]*10], [line[1], line[1] + dirVectX[1]*10],
            #           [line[2], line[2] + dirVectX[2]*10], color='r')
            # ax.plot([line[0], line[0] + dirVectYUpdated[0]*10], [line[1], line[1] + dirVectYUpdated[1]*10],
            #         [line[2], line[2] + dirVectYUpdated[2]*10], color='g')
            # ax.plot([line[0], line[0] + dirVectZ[0]*10], [line[1], line[1] + dirVectZ[1]*10],
            #         [line[2], line[2] + dirVectZ[2]*10], color='b')
            # ax.set_xlim([-50, 50])
            # ax.set_ylim([-20, 20])
            # ax.set_zlim([0, 100])
            # plt.pause(0.05)
            # plt.show()
            return transformMatrix

        def getImgToolPoses(self):
            self.imgPointCloud = self.get3DPts(self.leftMatchedPoints, self.rightMatchedPoints)
            for i in range(0, len(self.imgToolPts)):
                self.toolPts = self.imgToolPts[i]
                distance = self.distanceBetweenPoints(self.toolPts)
                self.toolLine = self.linePtCloud(self.imgPointCloud[i])
                self.setToolParameters(self.imgToolPts[i])
                self.updateToolLine()
                leftImageTools, rightImageTools = self.projecttoolsToImage()
                self.leftImageToolLines.append(leftImageTools)
                self.rightImageToolLines.append(rightImageTools)
                self.setToolPose()
                self.imgToolPose.append(self.toolPose)
            return

        def getRotationMatrixFromVector(self, directionVector):
            rotationMatrix, jacobianMatrix = cv2.Rodrigues(directionVector.astype(np.float))
            return rotationMatrix



        def getEulerAnglesFromDirection(self, directionVector):
            directionVector = directionVector / np.linalg.norm(directionVector)
            phi = math.atan2(directionVector[2], directionVector[0])
            theta = math.atan2(directionVector[1], np.sqrt(directionVector[0]**2 + directionVector[2]**2))
            return phi, theta

        def rotationMatrixFromEulerAngles(self, phi, theta):
            rotationMatrix = np.array([[math.sin(phi), math.cos(phi)* math.cos(theta), math.cos(phi)*math.sin(theta)],
                                        [0, math.sin(theta), -math.cos(theta)],
                                       [-math.cos(phi), math.sin(phi)*math.cos(theta), math.sin(phi)*math.sin(theta)]])
            return rotationMatrix

        def ptsProjectionOnLine(self, pts, line):
            dirVect = line[3:6]
            linePt = line[0:3]
            projectedPtOnLine = []
            if np.shape(np.shape(pts))[0] == 1:
                return projectedPtOnLine
            else:
                for i in range(0, np.shape(pts)[1]):
                    projectedPtOnLine.append(linePt + ((np.dot(linePt-pts[:, i], dirVect)/np.linalg.norm(dirVect)) * dirVect))
                projectedPtOnLine = np.transpose(np.asarray(projectedPtOnLine))
                return projectedPtOnLine

        def projectLineToImage(self, linePoints3D):
            leftImagePoints = []
            rightImagePoints = []
            if linePoints3D == []:
                return leftImagePoints, rightImagePoints
            else:
                leftImagePoints = cv2.projectPoints(np.transpose(linePoints3D), np.eye(3), np.array([0, 0, 0], dtype=float),
                                                     self.leftCameraMatrix, self.leftDistCoeffs)
                rightImagePoints = cv2.projectPoints(np.transpose(linePoints3D), cv2.Rodrigues(self.rightCameraRotation)[0],
                                                     self.rightCameraTranslation, self.rightCameraMatrix, self.rightDistCoeffs)
                leftImagePoints = leftImagePoints[0][:, 0, :]
                rightImagePoints = rightImagePoints[0][:, 0, :]
                invalidLeftImagePoints = []
                invalidRightImagePoints = []
                for i in range(0, len(leftImagePoints)):
                    if leftImagePoints[i, 0] > 960 or leftImagePoints[i, 1] > 540:
                        invalidLeftImagePoints.append(i)
                    if rightImagePoints[i, 0] > 960 or rightImagePoints[i, 1] > 540:
                        invalidRightImagePoints.append(i)
                leftImagePoints = np.delete(leftImagePoints, invalidLeftImagePoints, axis=0)
                rightImagePoints = np.delete(rightImagePoints, invalidRightImagePoints, axis=0)
            return leftImagePoints, rightImagePoints

        def projecttoolsToImage(self):
            toolCenterLinePoints = self.ptsProjectionOnLine(self.toolPts, self.toolLine)
            dist = self.distanceBetweenPoints(toolCenterLinePoints)
            toolLeftImage, toolRightImage = self.projectLineToImage(toolCenterLinePoints)
            return toolLeftImage, toolRightImage

        def distanceBetweenPoints(self, pts):
            if pts != []:
                distMat = np.zeros([np.shape(pts)[1]-1, 1])
                for i in range(1, np.shape(pts)[1]):
                    distMat[i-1] = np.sqrt((pts[0, 0] - pts[0, i])**2 + (pts[1, 0] - pts[1, i])**2 + (pts[2, 0] - pts[2, i])**2)
            else:
                distMat = []
            return distMat

        def removeOutliers(self, pts, line):
            distMat = np.abs(self.signedDistPtsFromLine(pts, line))
            inlierPts = np.transpose(np.array([pts[:, i] for i in range(0, len(distMat)) if distMat[i] < 8]))
            return inlierPts
