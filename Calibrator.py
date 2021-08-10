import numpy as np
import cv2
import glob2 as glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class Calibrator():
    def __init__(self, image_dir, nx, ny, debug=False):
        fnames = glob.glob("{}/*".format(image_dir))
        objpoints = []
        imgpoints = []

        # Coordinates of chessboard's corners in 3D
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        for f in fnames:
            img = mpimg.imread(f)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            ret, corners = cv2.findChessboardCorners(img, (nx, ny))
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)
            shape = (img.shape[1], img.shape[0])
            ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

            if not ret:
                raise Exception("无法校准摄像头")

    def undistort(self, img):
        # Convert to grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
