import numpy as np
import cv2

class PointPairsClass(object):

    def __init__(self, leftImagePts, rightImagePts, fundamentalMatrix):
        self.leftImagePts = leftImagePts
        self.rightImagePts = rightImagePts
        self.fundamentalMatrix = fundamentalMatrix
        self.leftImgMatchedPoints, self.rightImgMatchedPoints = self.getPointPairs(self.leftImagePts, self.rightImagePts)

    def getPointPairs(self, leftImgCurves, rightImgCurves):
        leftImgCurves, rightImgCurves = self.matchImgTools(leftImgCurves, rightImgCurves)
        leftImgMatchedPoints = []
        rightImgMatchedPoints = []
        for i in range(0, len(leftImgCurves)):
            lefttoolMatchedPoints = []
            righttoolMatchedPoints = []
            for j in range(0, len(leftImgCurves[i])):
                matchedLeftPoints, matchedRightPoints = self.getCorrespodingPoints(leftImgCurves[i][j],
                                                                                        rightImgCurves[i][j])
                lefttoolMatchedPoints.append(matchedLeftPoints)
                righttoolMatchedPoints.append(matchedRightPoints)
            leftImgMatchedPoints.append(lefttoolMatchedPoints)
            rightImgMatchedPoints.append(righttoolMatchedPoints)
        return leftImgMatchedPoints, rightImgMatchedPoints


    def getCurvesMean(self, imgCurvePts):
        imgCurvesMean = []
        for i in range(0, len(imgCurvePts)):
            meanList = []
            for j in range(0, len(imgCurvePts[i])):
                mean = np.mean(imgCurvePts[i][j], axis=0)
                meanList.append(mean)
            imgCurvesMean.append(meanList)
        return imgCurvesMean

    def matchImgTools(self, leftImgCurves, rightImgCurves):
        leftImgCurvesMean = self.getCurvesMean(leftImgCurves)
        rightImgCurvesMean = self.getCurvesMean(rightImgCurves)
        leftImgMean=[]
        rightImgMean=[]
        distMat = []
        lefttoolMean = np.mean(leftImgCurvesMean[0], axis=0)
        for i in range(0, len(leftImgCurvesMean)):
            righttoolMean = np.mean(rightImgCurvesMean[i], axis=0)
            dist = np.linalg.norm(lefttoolMean-righttoolMean)
            distMat.append(dist)
        if distMat[0] > distMat[1]:
            rightToolsLabel = [1, 0]
            rightImgCurves = [rightImgCurves[i] for i in rightToolsLabel]
            rightImgCurvesMean = [rightImgCurvesMean[i] for i in rightToolsLabel]

        for i in range(0, len(leftImgCurvesMean)):
            leftImgCurves[i], leftImgCurvesMean[i] = self.sortToolCurves(leftImgCurves[i], leftImgCurvesMean[i])
            rightImgCurves[i], rightImgCurvesMean[i] = self.sortToolCurves(rightImgCurves[i], rightImgCurvesMean[i])
            leftImgCurves[i], rightImgCurves[i] = self.matchToolCurves(leftImgCurves[i], rightImgCurves[i])
        return leftImgCurves, rightImgCurves

    def matchToolCurves(self, leftToolCurves, rightToolCurves):
        sizeLeft = len(leftToolCurves)
        sizeRight = len(rightToolCurves)
        if sizeRight != sizeLeft:
            leftToolCurves = leftToolCurves[:min(sizeLeft, sizeRight)]
            rightToolCurves = rightToolCurves[:min(sizeLeft, sizeRight)]
        return leftToolCurves, rightToolCurves

    def minDistIdx(self, pts1, pts2):
        distMat = np.empty([len(pts1), len(pts2)])
        pts2MinIdx = np.empty([len(pts1), 1])
        for i in range(len(pts1)):
            for j in range(len(pts2)):
                distMat[i, j] = np.linalg.norm(pts1[i]-pts2[j])
            pts2MinIdx[i] = np.argmin(distMat[i])
        return pts2MinIdx

    def sortToolCurves(self, toolCurves, meanToolCurves): #TODO: define order of functions as you proceed
        distMat = []
        for i in range(0, len(toolCurves)):
            if i==0:
                distMat.append(0)
            if i>0:
                dist = meanToolCurves[i][0]-meanToolCurves[0][0]
                distMat.append(dist)
        sortedIdx = np.argsort(distMat)
        toolCurves = [toolCurves[i] for i in sortedIdx]
        meanToolCurves = [meanToolCurves[i] for i in sortedIdx]
        return toolCurves, meanToolCurves

    def getCorrespodingPoints(self, leftCurves, rightCurves):
        if leftCurves.size and rightCurves.size:
            leftCurves = cv2.convertPointsToHomogeneous(leftCurves)
            rightCurves = cv2.convertPointsToHomogeneous(rightCurves)
            # distMat = np.empty([len(leftCurves),len(rightCurves)])
            distMatCV = np.empty([len(leftCurves),len(rightCurves)])
            # minDistMat = np.empty([len(leftCurves), 2])
            for i in range(0, len(leftCurves)):
                # lineRight = np.matmul(self.fundamentalMatrix, leftCurves[i])
                lineRightCV = cv2.computeCorrespondEpilines(leftCurves[i], 1, self.fundamentalMatrix)
                for j in range(0, len(rightCurves)):
                    # distMat[i, j] = np.abs(np.matmul(rightCurves[j], lineRight))
                    distMatCV[i, j] = np.abs(np.matmul(rightCurves[j], lineRightCV[0,0,:]))
            leftIdx = []
            rightIdx = []
            for i in range(len(leftCurves)):
                potentialRightIdx = np.argmin(distMatCV[i, :])
                potentialLeftIdx = np.argmin(distMatCV[:, potentialRightIdx])
                if potentialLeftIdx == i and distMatCV[i, potentialRightIdx] < 0.5:
                    leftIdx.append(potentialLeftIdx)
                    rightIdx.append(potentialRightIdx)
            if leftIdx and rightIdx:
                matchedLeftPoints = cv2.convertPointsFromHomogeneous(np.array([leftCurves[leftIdx[i]] for i in
                                                                               range(0, len(leftIdx))])).reshape([1, len(leftIdx), 2])
                matchedRightPoints = cv2.convertPointsFromHomogeneous(np.array([rightCurves[rightIdx[i]] for i in
                                                                                range(0, len(rightIdx))])).reshape([1, len(leftIdx), 2])
                matchedLeftPointsCV, matchedRightPointsCV = cv2.correctMatches(self.fundamentalMatrix, matchedLeftPoints
                                                                           , matchedRightPoints)
                matchedLeftPointsCV = matchedLeftPointsCV.reshape([len(leftIdx), 2])
                matchedRightPointsCV = matchedRightPointsCV.reshape(len(leftIdx), 2)
            else:
                matchedLeftPointsCV = []
                matchedRightPointsCV = []
        else:
            matchedLeftPointsCV = []
            matchedRightPointsCV = []
        return matchedLeftPointsCV, matchedRightPointsCV
