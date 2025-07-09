import time
import cv2
import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


from cam_calibration import HandInEyeCalibration
from CR_robot import CR_ROBOT, CR_bringup
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType

# 放置的位置
place_position = [-650, -200, 400]
# 初始位置
home_pose = [-300, 0, 500, 180, -5, 45]
# x，y是变量，只用z，rx, ry, rz
height_pose = [-400, 0,345, 180, -5, 45]

capture = cv2.VideoCapture(1)

if __name__ == '__main__':

    tcp_host_ip = "192.168.5.1"
    tcp_port = 30003
    tcp_port2 = 29999

    robot = CR_ROBOT(tcp_host_ip, tcp_port)
    robot_init = CR_bringup(tcp_host_ip, tcp_port2)
    robot_init.EnableRobot()

    # Load model
    device = select_device('')
    weights = "weights/weights/best.pt"
    dnn = False
    data = "data/1234.yaml"
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size((640, 480), s=stride)  # check image size

    model.warmup()  # warmup

    capture = cv2.VideoCapture(1)

    while True:
        start_time = time.time()
        detected_objects = []
        detection_time_limit = 1  # 设置检测的时间限制为1秒

        while True:
            ret, frame = capture.read()
            img0 = frame
            img = letterbox(frame)[0]
            img = frame.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            im = torch.from_numpy(img).to(device)
            im = im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

            # 检测图像
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred)
            det = pred[0]

            if len(det):
                annotator = Annotator(frame, line_width=3, example=str(names))
                for *xyxy, conf, cls in det:
                    c = int(cls)
                    label = names[c]
                    annotator.box_label(xyxy, str(label), color=colors(c, True))

                # 获取检测框并记录面积最大目标
                largest_area = 0
                largest_object = None
                for i, (*xyxy, conf, cls) in enumerate(det):
                    x1, y1, x2, y2 = xyxy
                    area = (x2 - x1) * (y2 - y1)
                    if area > largest_area:
                        largest_area = area
                        largest_object = det[i]

                detected_objects.append(largest_object)

                im0 = annotator.result()
                cv2.imshow('frame', im0)

            # 停止1秒后选择检测框最大的物体
            if time.time() - start_time > detection_time_limit:
                if detected_objects:
                    largest_object = max(detected_objects, key=lambda obj: (obj[2] - obj[0]) * (obj[3] - obj[1]))
                    x1, y1, x2, y2 = largest_object[:4]
                    x_camera, y_camera = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    print("中间点坐标：({}, {})".format(x_camera, y_camera))

                    calibrator = HandInEyeCalibration()
                    robot_x, robot_y = calibrator.get_points_robot(x_camera, y_camera)
                    print("机械臂坐标 (x, y):", robot_x, robot_y)

                    '''-------------------抓取物体------------'''
                    robot.MovJ(int(home_pose[0]), int(home_pose[1]), int(home_pose[2]),
                               home_pose[3], home_pose[4], home_pose[5])
                    time.sleep(3)
                    robot.MovJ(int(robot_x)-10, int(robot_y), int(height_pose[2] + 100),
                               height_pose[3], height_pose[4], height_pose[5])
                    time.sleep(3)
                    robot.MovJ(int(robot_x)-10, int(robot_y), int(height_pose[2]),
                               height_pose[3], height_pose[4], height_pose[5])
                    time.sleep(3)
                    robot_init.ToolDOExecute(2, 0)
                    time.sleep(3)
                    robot.MovJ(int(robot_x)-10, int(robot_y), int(height_pose[2] + 100),
                               height_pose[3], height_pose[4], height_pose[5])
                    time.sleep(3)
                    robot.MovJ(int(place_position[0]), int(place_position[1]), int(place_position[2]),
                               height_pose[3], height_pose[4], height_pose[5])
                    time.sleep(3)
                    robot_init.ToolDOExecute(2, 1)
                    time.sleep(3)
                    robot.MovJ(int(home_pose[0]), int(home_pose[1]), int(home_pose[2]),
                               home_pose[3], home_pose[4], home_pose[5])
                    '''------------------------------------------------------------------------------'''

                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("抓取完成，等待10秒进入下一次抓取...")
        time.sleep(10)  # 每次循环抓取操作后等待10秒

    capture.release()
    cv2.destroyAllWindows()
