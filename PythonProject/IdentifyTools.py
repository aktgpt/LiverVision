import cv2
import numpy as np
import math

class IdentifyTools:

    def __init__(self, imageToolCurves, imageToolCenterLine):
        self.imageToolCurves = imageToolCurves
        self.imageToolCenterLine = imageToolCenterLine
        self.labelCurves()
        self.tool1Ratio = np.array([1, 2, 1.5, 1, 1.5])
        self.tool2Ratio = np.array([1.5, 1, 1, 2, 1.5])

    def findPointsOnToolCenterline(self, toolCurves, toolLine):
        pointsCenterLine = []
        for i in range(0, len(toolCurves)):
            errorMat = np.zeros([len(toolCurves[i]), 1])
            for j in range(0, len(toolCurves[i])):
                errorMat[j] = abs(toolCurves[i][j, 1] - (toolLine[0]*toolCurves[i][j, 0]) - toolLine[1])
            pointsCenterLine.append(toolCurves[i][errorMat.argmin(), :])
        return pointsCenterLine

    def sortToolCurves(self, toolCurves, pointsCenterLine):
        distXaxis = []
        for i in range(0, len(pointsCenterLine)):
            if i==0:
                distXaxis.append(0)
            else:
                distXaxis.append(pointsCenterLine[i][0] - pointsCenterLine[0][0])
        sortedIdx = np.argsort(distXaxis, axis=0)
        toolCurves = [toolCurves[i] for i in sortedIdx]
        pointsCenterLine = [pointsCenterLine[i] for i in sortedIdx]
        return toolCurves, pointsCenterLine


    def getCurvesDistance(self, pointsToolCenterLine):
        distMat = []
        for i in range(1, len(pointsToolCenterLine)):
            distMat.append(np.linalg.norm(pointsToolCenterLine[i] - pointsToolCenterLine[i-1]))
        return distMat

    def getCurveRatio(self, distanceBetweenCurves):
        ratioMat = []
        mindistIdx = np.argmin(distanceBetweenCurves)
        for i in range(0, len(distanceBetweenCurves)):
            if i == mindistIdx:
                ratioMat.append(1)
            else:
                ratioMat.append(round(distanceBetweenCurves[i] / distanceBetweenCurves[mindistIdx] * 2) / 2)
        return ratioMat

    def findToolIdentity(self, curvesRatio):

        pass


    def labelCurves(self):
        for i in range(len(self.imageToolCurves)):
            self.toolCenterLine = self.findPointsOnToolCenterline(self.imageToolCurves[i], self.imageToolCenterLine[i])
            self.imageToolCurves[i], toolCenterLine = self.sortToolCurves(self.imageToolCurves[i], self.toolCenterLine)
            distanceBWCurves = self.getCurvesDistance(toolCenterLine)
            distanceratio = self.getCurveRatio(distanceBWCurves)
            print(distanceratio)
        return distanceratio



