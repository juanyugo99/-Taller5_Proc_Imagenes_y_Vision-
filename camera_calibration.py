import os
import cv2
import glob
import json
import random
import numpy as np


if __name__ == '__main__':

    # calibtarion pattern parameters
    xdim = 7
    ydim = 6
    real_square_size = 23  # in mm

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ydim * xdim, 3), np.float32)
    objp[:, :2] = np.mgrid[0:xdim, 0:ydim].T.reshape(-1, 2)
    objp *= real_square_size/1000   # calibrate point to real size dimension

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Search all calibration images

    select = int(input('Select which camera do you want to calibrate? laptop (0) or Phone(1): '))
    if select == 0:
        path = 'C:/Users/Juan Pablo/Im_Procesamiento/calibration_laptop_images' # put here your laptop camera images
    else:
        path = 'C:/Users/Juan Pablo/Im_Procesamiento/calibration_phone_images'  # put here your phone camera images

    path_file = os.path.join(path, '*.jpg') # here change your images format (.png, .jpg, .jpeg, .bmp, etc...)

    images = glob.glob(path_file)
    count_detected = 0
    count_all = 0
    detected_images = []

    for fname in images:
        count_all += 1
        img = cv2.imread(fname)
        img = cv2.resize(img, tuple([640, 480]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (xdim, ydim),
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)
        if ret == True:
            count_detected += 1
            detected_images.append(count_all)

            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (xdim, ydim), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(70)

    cv2.destroyAllWindows()

    print('images detected: {} / {} '.format(count_detected, len(images)))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print('intrinsic camera parameters are:\n{}'.format(mtx))
    print('ditorition camera parameters are:\n {}'.format(dist))

    # Export parameters
    file_name = 'calibration.json'
    json_file = os.path.join(path, file_name)

    data = {
        'K': mtx.tolist(),
        'distortion': dist.tolist(),
        'tilt': [0, 30],
        'pan': [5, 0],
        'd': [2.0, 3.0],
        'h': [1.0, 2.0]
    }

    with open(json_file, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=1, ensure_ascii=False)

    with open(json_file) as fp:
        json_data = json.load(fp)

    # reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: {}".format(mean_error / len(objpoints)))

    # undistortion
    selector = random.choice(detected_images)
    path_file = images[selector]
    img = cv2.imread(path_file)
    img = cv2.resize(img, tuple([640, 480]))
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    if True:
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    else:
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imshow('distorted', img)
    cv2.imshow('calibresult', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()