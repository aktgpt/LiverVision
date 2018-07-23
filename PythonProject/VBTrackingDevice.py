import StereoCameraCalibration
import toolsInImageClass as toolsInImage
import PointPairsClass as pointPair
import poseEstimateClass as poseEstimate


class VBTrackingDevice:

    def __init__(self):
        pass

    def initialize(self, calibration):
        self.fundamentalMatrix = calibration['FundamentalMatrix']
        self.leftProjectionMatrix = calibration['LeftProjectionMatrix']
        self.rightProjectionMatrix = calibration['RightProjectionMatrix']

    def track(self, inputLeftImage, inputRightImage):
        leftTools = toolsInImage.InstrumentsInImage(inputLeftImage, 2)
        self.leftImageTools = leftTools.getImageTools()

        rightTools = toolsInImage.InstrumentsInImage(inputRightImage, 2)
        self.rightImageTools = rightTools.getImageTools()

        correspondingPoints = pointPair.PointPairsClass(self.leftImageTools, self.rightImageTools, self.fundamentalMatrix)
        self.matchedLeftCurves = correspondingPoints.leftImgMatchedPoints
        self.matchedRightCurves = correspondingPoints.rightImgMatchedPoints

    def getTransformations(self):
        pose = poseEstimate.poseEstimateClass(self.matchedLeftCurves, self.matchedRightCurves,
                                              self.leftProjectionMatrix, self.rightProjectionMatrix)
        pose.getImgToolPoses()
        self.toolPoses = pose.imgToolPose

        return self.toolPoses