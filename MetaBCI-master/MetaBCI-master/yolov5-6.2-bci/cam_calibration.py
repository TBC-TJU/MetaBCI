import numpy as np
import cv2

# 通过九点标定获取的圆心相机坐标
STC_points_camera = np.array([

    [249, 70],
    [252, 132],
    [253, 191],
    [322, 190],
    [321, 131],
    [319, 68],
    [386, 67],
    [389, 129],
    [391, 189],
])
# 通过九点标定获取的圆心机械臂坐标
STC_points_robot = np.array([
    [-465, 53],
    [-539, 53],
    [-610, 55],
    [-612, -27],
    [-541, -29],
    [-467, -27],
    [-469, -107],
    [-543, -109],
    [-614, -109],
])


# 手眼标定方法
class HandInEyeCalibration:

    def get_m(self, points_camera, points_robot):
        """
        取得相机坐标转换到机器坐标的仿射矩阵
        :param points_camera:
        :param points_robot:
        :return:
        """
        # 确保两个点集的数量级不要差距过大，否则会输出None
        m, _ = cv2.estimateAffine2D(points_camera, points_robot)
        return m

    def get_points_robot(self, x_camera, y_camera):
        """
        相机坐标通过仿射矩阵变换取得机器坐标
        :param x_camera:
        :param y_camera:
        :return:
        """
        m = self.get_m(STC_points_camera, STC_points_robot)
        robot_x = (m[0][0] * x_camera) + (m[0][1] * y_camera) + m[0][2]
        robot_y = (m[1][0] * x_camera) + (m[1][1] * y_camera) + m[1][2]
        return robot_x, robot_y