import os
import cv2
import json
import numpy as np
from camera_model import *

if __name__ == '__main__':

    # path of JSON calibration camera parameters
    path = 'C:/Users/Juan Pablo/Im_Procesamiento/calibration_phone_images/'
    json_file = os.path.join(path, 'calibration.json')

    # Extract intrinsic and extrinsic camera parameters
    with open(json_file, "r") as fp:
        json_data = json.load(fp)
        K = np.array(json_data['K']).astype(float)
        distortion = np.array(json_data['distortion']).astype(float)
        tilt = json_data['tilt']
        pan = json_data['pan']
        d = json_data['d']
        h = json_data['h']

    # resolution of image projection
    width = 1280
    height = 720

    # extrinsic parameters
    select = int(input('Select Camera perspective (0/1): '))
    if select == 0:
        R = set_rotation(tilt[0], pan[0], skew=0)
        t = np.array([0, -d[1], h[1]])
    else:
        R = set_rotation(tilt[1], pan[1], skew=0)
        t = np.array([0, -d[1], h[1]])

    # create camera
    camera = projective_camera(K, width, height, R, t)
    # Create Cube
    sizec = 1 # size of cube in meters
    axis = np.float32([[0, 0, 0], [0, sizec, 0], [sizec, sizec, 0], [sizec, 0, 0],
                       [0, 0, -sizec], [0, sizec, -sizec], [sizec, sizec, -sizec], [sizec, 0, -sizec]])

    im_pt = projective_camera_project(axis, camera)
    im_pt = np.int32(im_pt).reshape(-1, 2)
    image_projective = 255 * np.ones(shape=[camera.height, camera.width, 3], dtype=np.uint8)

    color = (0, 0, 255)
    color1 = (0, 255, 0)

    for i, j in zip(range(4), range(4, 8)):
        image_projective = cv2.line(image_projective, tuple(im_pt[i]), tuple(im_pt[j]), color, 2)

    image_projective = cv2.drawContours(image_projective, [im_pt[:4]], -1, color1, 2)
    image_projective = cv2.drawContours(image_projective, [im_pt[4:]], -1, color, 2)

    cv2.imshow("Cube", image_projective)
    cv2.waitKey(0)