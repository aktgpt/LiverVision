import numpy as np
import cv2
import glob
import os

class StereoCalibration:

    def __init__(self, filespath, calibrationFromVideo, calibrationFromImages, loadFromSavedFile):
        # if external calibration is required
        self.calibrationFromVideo = calibrationFromVideo
        self.calibrationFromImages = calibrationFromImages
        self.loadFromSavedFile = loadFromSavedFile

        self.imagesForCalibration = []
        self.filesPath = filespath

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((9*6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)*25.25

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        if self.calibrationFromVideo:
            self.getImagesFromCamera()
            self.readImages()
            self.getCalibrationParameters()
            self.saveCalibrationParameters()
        if self.calibrationFromImages:
            self.getImagesFromFolder()
            self.readImages()
            self.getCalibrationParameters()
            self.saveCalibrationParameters()
        if self.loadFromSavedFile:
            self.loadCalibrationParameters()

    def getImagesFromCamera(self):
        # get images after every 0.5 seconds and save them as list of left and right images
        # self.imagesForCalibration = []
        pass

    def getImagesFromFolder(self):
        images_right = glob.glob(self.filesPath + 'right/*.JPG')
        images_left = glob.glob(self.filesPath + 'left/*.JPG')
        images_left.sort()
        images_right.sort()
        imgs_l = []
        imgs_r = []
        for i in range(len(images_left)):
            imgs_l.append(cv2.imread(images_left[i]))
            imgs_r.append(cv2.imread(images_right[i]))
        self.imagesForCalibration.append(imgs_l)
        self.imagesForCalibration.append(imgs_r)

    def readImages(self):
        for i in range(len(self.imagesForCalibration[0])):
            img_l = self.imagesForCalibration[0][i]
            img_r = self.imagesForCalibration[1][i]

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

            if ret_l is True and ret_r is True:
                # If found, add object points, image points (after refining them)
                self.objpoints.append(self.objp)

                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

            self.img_shape = gray_l.shape[::-1]

    def getCalibrationParameters(self):
        rt, M1, d1, r1, t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, self.img_shape, None, None)
        rt, M2, d2, r2, t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, self.img_shape, None, None)

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, M1, d1, M2,
            d2, self.img_shape,
            criteria=stereocalib_criteria, flags=flags)

        extrinsicParams = cv2.stereoRectify(M1, d1, M2, d2, self.img_shape, R, T)

        self.StereoParameters = dict([('LeftCameraMatrix', M1),
                                      ('RightCameraMatrix', M2),
                                      ('LeftDistortionCoefficients', d1),
                                      ('RightDistortionCoefficients', d2),
                                      ('RightCameraRotation', R),
                                      ('RightCameraTranslation', T),
                                      ('FundamentalMatrix', F),
                                      ('LeftProjectionMatrix', extrinsicParams[2]),
                                      ('RightProjectionMatrix', extrinsicParams[3])])

    def saveCalibrationParameters(self):
        np.savez("StereoParameters",
                 LeftCameraMatrix=self.StereoParameters['LeftCameraMatrix'],
                 RightCameraMatrix=self.StereoParameters['RightCameraMatrix'],
                 LeftDistortionMatrix=self.StereoParameters['LeftDistortionCoefficients'],
                 RightDistortionMatrix=self.StereoParameters['RightDistortionCoefficients'],
                 RightCameraRotation=self.StereoParameters['RightCameraRotation'],
                 RightCameraTranslation=self.StereoParameters['RightCameraTranslation'],
                 FundamentalMatrix=self.StereoParameters['FundamentalMatrix'],
                 LeftProjectionMatrix=self.StereoParameters['LeftProjectionMatrix'],
                 RightProjectionMatrix=self.StereoParameters['RightProjectionMatrix']),

    def loadCalibrationParameters(self):
        if os.path.isfile(self.filesPath + "StereoParameters.npz"):
            calibrationParameters = np.load("StereoParameters.npz")
            self.StereoParameters = dict([('LeftCameraMatrix', calibrationParameters['LeftCameraMatrix']),
                                          ('RightCameraMatrix', calibrationParameters['RightCameraMatrix']),
                                          ('LeftDistortionMatrix', calibrationParameters['LeftDistortionCoefficients']),
                                          ('RightDistortionMatrix', calibrationParameters['RightDistortionCoefficients']),
                                          ('RightCameraRotation', calibrationParameters['RightCameraRotation']),
                                          ('RightCameraTranslation', calibrationParameters['RightCameraTranslation']),
                                          ('FundamentalMatrix', calibrationParameters['FundamentalMatrix']),
                                          ('LeftProjectionMatrix', calibrationParameters['LeftProjectionMatrix']),
                                          ('RightProjectionMatrix', calibrationParameters['RightProjectionMatrix'])])
        else:
            print("Stereo Parameters not found")

