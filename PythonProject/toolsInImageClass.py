import cv2
import numpy as np
import sklearn
import math
import scipy
import sklearn.cluster


class InstrumentsInImage:   #TODO: DetectToolInImage

    # TODO: Define class variable as left image and right image in input parser
    def __init__(self, inputRGBImage, numTools):
        self.inputRGBImage = inputRGBImage
        self.numTools = numTools
        self.toolCenterLine = np.zeros([numTools, 2])
        self.imgToolContour = []

    def getImageTools(self):
        # leftImage, rightImage = self.separateImages(self.inputStereoImage)

        self.ImgTools = self.getToolCurves(self.greenMask(self.inputRGBImage))
        # self.rightImgTools = self.getToolCurves(self.greenMask(rightImage))
        return self.ImgTools

    # def separateImages(self, inputStereoImage):
    #     leftImage = inputStereoImage[0:540, 480:1440, :]
    #     rightImage = inputStereoImage[541:960, 480:1440, :]
    #     return leftImage, rightImage

    def greenMask(self, inputRGBImage):
        imageBlur = cv2.GaussianBlur(inputRGBImage, (5, 5), 0) #TODO: remove variables which are used once
        imageHSV = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2HSV)

        lowerThresholdGreen = np.array([25, 25, 0]) #TODO: define complete name of variables
        upperThresholdGreen = np.array([130, 255, 255]) #TODO: define complete name of variables
        imageMask = cv2.inRange(imageHSV,lowerThresholdGreen,upperThresholdGreen)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        imageMaskFiltered = cv2.morphologyEx(imageMask, cv2.MORPH_CLOSE, kernel)
        return imageMaskFiltered

    def getLargestContours(self, inputBinImage):
        img, imgContours, hierarchy = cv2.findContours(inputBinImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        thresholdArea = 400
        imgContoursLargest = []
        # areaCon = []
        for i in range(0, len(imgContours)):
            area = cv2.contourArea(imgContours[i])
            if area > thresholdArea and hierarchy[0][i][2] < 0:
                imgContours[i] = np.reshape(imgContours[i], [np.shape(imgContours[i])[0], 2])
                imgContoursLargest.append(np.asarray(imgContours[i]))
                # areaCon.append(area)

        return imgContoursLargest

    def groupTools(self, imgContours, numTools):
        orientMat = np.empty([len(imgContours), 3])
        for i in range(0, len(imgContours)):
            M = cv2.moments(imgContours[i])
            orientMat[i] = self.getCenterandOrientation(M)

        kmeansLabels = sklearn.cluster.KMeans(n_clusters=numTools).fit(orientMat).labels_
        imgToolContours = []
        centerLine = []
        global toolParams, toolContours #TODO: Avoid defining global variable, change to class variable
        for i in range(0, numTools):
            idx = np.where(kmeansLabels == i)
            if len(idx[0]) >= 2:
                toolParams = orientMat[idx[0]]
                x = toolParams[0:len(idx[0]), 0]
                y = toolParams[0:len(idx[0]), 1]
                z = np.polyfit(x, y, 1)
                self.toolCenterLine[i, :] = z
                toolParams[:, 2] = z[0]
                toolContours = []
                for i1 in range(0, len(idx[0])):
                    toolContours.append(imgContours[idx[0][i1]])
            centerLine.append(np.array(toolParams))
            imgToolContours.append(toolContours)
        centerLine = np.asarray(centerLine)

        return imgToolContours, centerLine

    def getCenterandOrientation(self, moment): #TODO: define complete name of moment variables
        cx = moment['m10'] / moment['m00']
        cy = moment['m01'] / moment['m00']
        mu11 = moment['m11'] / moment['m00'] - cx * cy
        mu02 = moment['m02'] / moment['m00'] - cy ** 2
        mu20 = moment['m20'] / moment['m00'] - cx ** 2
        orientation = 0.5 * np.arctan((2*mu11)/(mu20-mu02))
        return [cx, cy, orientation]

    def findOrientation(self, toolParams,idx):
        x = toolParams[0:len(idx[0]), 0]
        y = toolParams[0:len(idx[0]), 1]
        z = np.polyfit(x, y, 1)

        orientation = z[0]
        return orientation

    def getLinesandCurves(self, contour, centerLine):
        orientation = centerLine[2]
        center = centerLine[:2]
        transformedData = self.transformContour(contour, center, orientation)

        quadMat = self.getQuadrantandDist(transformedData)
        crnrIdx = self.findCornerIdx(quadMat)

        corners = contour[crnrIdx]
        rightpartIdx = [i for i in range(len(quadMat)) if quadMat[i][0] == 1 or quadMat[i][0] == 4]
        leftpartIdx = [i for i in range(len(quadMat)) if quadMat[i][0] == 2 or quadMat[i][0] == 3]
        rightpartContour = contour[rightpartIdx]
        leftpartContour = contour[leftpartIdx]

        contourLine = []
        contourCurve = []
        rightlinePtsIdx, rightcurvePtsIdx = self.idxLineOrCurve(rightpartContour, corners)
        leftlinePtsIdx, leftcurvePtsIdx = self.idxLineOrCurve(leftpartContour, corners)

        contourCurve.append(rightpartContour[rightcurvePtsIdx])
        contourCurve.append(leftpartContour[leftcurvePtsIdx])

        contourLine.append(rightpartContour[rightlinePtsIdx])
        contourLine.append(leftpartContour[leftlinePtsIdx])

        return contourLine, contourCurve

    def getQuadrantandDist(self, pts):
        # pts = np.transpose(np.reshape(pts, [len(pts), 2]))
        dtype = [('Quadrant', int), ('DistanceFromOrigin', float)]
        quadMat = np.zeros(np.shape(pts)[1], dtype=dtype)
        for i in range(0, np.shape(pts)[1]):
            if pts[0, i] > 0 and pts[1, i] > 0:
                quadMat[i] = (1, np.linalg.norm(pts[:, i]))
            elif pts[0, i] < 0 and pts[1, i] > 0:
                quadMat[i] = (2, np.linalg.norm(pts[:, i]))
            elif pts[0, i] < 0 and pts[1, i] < 0:
                quadMat[i] = (3, np.linalg.norm(pts[:, i]))
            elif pts[0, i] > 0 and pts[1, i] < 0:
                quadMat[i] = (4, np.linalg.norm(pts[:, i]))
        return quadMat

    def findCornerIdx(self, quadMat):
        idx = np.argsort(quadMat, order=['Quadrant', 'DistanceFromOrigin'])
        for i in range(0, len(quadMat)):
            if quadMat[idx[i]][0] == 2 and quadMat[idx[i-1]][0] == 1:
                trIdx = idx[i-1]
            if quadMat[idx[i]][0] == 3 and quadMat[idx[i-1]][0] == 2:
                tlIdx = idx[i-1]
            if quadMat[idx[i]][0] == 4 and quadMat[idx[i-1]][0] == 3:
                blIdx = idx[i-1]
        brIdx = idx[len(quadMat)-1]
        return np.array([trIdx, tlIdx, blIdx, brIdx])

    def transformContour(self, contour, center, orientation):
        theta = -(math.atan(orientation))
        RMat = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])

        transformedData = np.empty([2, len(contour)])
        for i in range(0, len(contour)):
            transformedData[:, i] = np.matmul(RMat, contour[i, :] - center)

        return transformedData

    def idxLineOrCurve(self, contour, corners):
        threshDist = 3
        isLinePt = np.zeros([1, len(contour)], dtype=bool)
        for i in range(0, 2):
            point1 = corners[2 * i, :]
            point2 = corners[2 * i + 1, :]
            kLine = point2 - point1
            kLineNorm = kLine / np.linalg.norm(kLine)
            normVector = np.asarray([-kLineNorm[1], kLineNorm[0]])
            distance = abs((contour - point1).dot(normVector))
            isInlierLine = distance <= threshDist
            isLinePt = scipy.logical_or(isLinePt, isInlierLine)
        linePtsIdx = np.where(isLinePt == True)[1]
        curvePtsIdx = np.where(isLinePt == False)[1]
        return linePtsIdx, curvePtsIdx

    def getToolCurves(self, inputBinImage):
        imgContoursLargest = self.getLargestContours(inputBinImage)
        imgToolContours, centerLine = self.groupTools(imgContoursLargest, self.numTools)
        self.imgToolContour = imgToolContours
        imgToolCurves = []
        # imgToolIdx = []
        for i in range(0, len(imgToolContours)):
            toolCurves = []
            for j in range(0, len(imgToolContours[i])):
                contourLine, contourCurve = self.getLinesandCurves(imgToolContours[i][j], centerLine[i][j, :])
                toolCurves.append(contourCurve[0])
                toolCurves.append(contourCurve[1])
            # toolCurveIdx = self.labelToolCurves(toolCurves)
            imgToolCurves.append(toolCurves)
            # imgToolIdx.append(toolCurveIdx)
        return imgToolCurves

    def getToolsBoundingBox(self):
        for i in range(len(self.leftImgTools)):
            toolPoints = np.empty([0, 2])
            for j in range(len(self.leftImgTools[i])):
                toolPoints = np.vstack((toolPoints, self.leftImgTools[i][j]))
                rect = cv2.minAreaRect(np.reshape(toolPoints, [len(toolPoints), 1, 2]))
        pass

    def ptsOnToolLine(self, toolContour, centerLine):
        for i in range(0, len(toolContour)):
            errorMat = np.zeros([len(toolContour[i]), 1])
            for j in range(0, len(toolContour[i])):
                errorMat[i] = toolContour[i][j][2] - toolContour[i][j][1] * centerLine[0] - centerLine[1]
        return

    def getCurveRatio(self, toolCurves, toolLine):
        pass




