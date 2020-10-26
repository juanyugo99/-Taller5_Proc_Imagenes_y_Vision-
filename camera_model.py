import numpy as np


class pinhole_camera:
    def __init__(self, K, width, height):
        self.K = K
        self.width = width
        self.height = height


def pinhole_camera_project(p_3D, camera):

    p_2D = np.matmul(camera.K, p_3D.T)

    for i in range(2):
        p_2D[i, :] /= p_2D[2, :]

    p_2D = p_2D[:2, :]
    p_2D = p_2D.transpose()
    p_2D = p_2D.astype(int)

    return p_2D


class projective_camera:
    def __init__(self, K, width, height, R, t):
        self.K = K
        self.width = width
        self.height = height
        self.R = R
        self.t = t


def projective_camera_project(p_3D, camera):

    p_3D_ = np.copy(p_3D)
    for i in range(3):
        p_3D_[:, i] = p_3D_[:, i] - camera.t[i]

    p_3D_cam = np.matmul(camera.R, p_3D_.T)
    p_2D = np.matmul(camera.K, p_3D_cam)

    for i in range(2):
        p_2D[i, :] /= p_2D[2, :]

    p_2D = p_2D[:2, :]
    p_2D = p_2D.transpose()
    p_2D = p_2D.astype(int)

    return p_2D


def set_rotation(tilt, pan=0, skew=0):

    R = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    theta_x = tilt * np.pi / 180
    theta_y = skew * np.pi / 180
    theta_z = pan * np.pi / 180

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])

    R_ = np.matmul(np.matmul(Rz, Ry), Rx)
    R_new = np.matmul(R, R_)

    return R_new