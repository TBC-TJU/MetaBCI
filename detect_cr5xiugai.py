import time
import cv2
import numpy as np
import torch
import socket

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from cam_calibration import HandInEyeCalibration
from CR_robot import CR_ROBOT, CR_bringup
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType

# --------------------参数设置----------------------
place_position = [-750, -230, 400]
home_pose = [-300, 0, 500, 180, -5, 45]
height_pose = [-400, 0, 222, 180, -5, 45]

# 标签编号到目标类的映射（与你的数据集类别对应）
label_to_object = {
    1: '1',
    2: '2',
    3: '3',
    4: '4'
}

# --------------------初始化------------------------
local_addr = ('192.168.0.172', 8000)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind(local_addr)

capture = cv2.VideoCapture(1)
device = select_device('')
weights = "weights/weights/best.pt"
data = "data/1234.yaml"
model = DetectMultiBackend(weights, device=device, data=data, dnn=False)
stride, names, *_ = model.stride, model.names
imgsz = check_img_size((640, 480), s=stride)
model.warmup()

# 机械臂初始化
tcp_host_ip = "192.168.5.1"
tcp_port = 30003
tcp_port2 = 29999
robot = CR_ROBOT(tcp_host_ip, tcp_port)
robot_init = CR_bringup(tcp_host_ip, tcp_port2)
robot_init.EnableRobot()

while True:
    print("等待接收标签编号...")
    recv_data, addr = udp_socket.recvfrom(4)
    label_id = int.from_bytes(recv_data, byteorder='little', signed=False)
    print(f"接收到标签编号：{label_id}")

    if label_id not in label_to_object:
        print(f"未知标签 {label_id}，忽略！")
        continue

    target_label = label_to_object[label_id]
    print(f"目标物体是：{target_label}")

    start_time = time.time()
    detection_time_limit = 2
    detected_target = None

    while time.time() - start_time < detection_time_limit:
        ret, frame = capture.read()
        if not ret:
            continue

        img0 = frame
        img = letterbox(frame)[0]
        img = frame.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device)
        im = im.float()
        im /= 255
        if im.ndim == 3:
            im = im.unsqueeze(0)

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred)
        det = pred[0]

        annotator = Annotator(frame, line_width=3, example=str(names))

        if len(det):
            for *xyxy, conf, cls in det:
                c = int(cls)
                label = str(names[c])
                annotator.box_label(xyxy, label, color=colors(c, True))

                if label == target_label:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detected_target = (x1, y1, x2, y2)
                    break

        im0 = annotator.result()
        cv2.imshow('frame', im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if detected_target:
        x1, y1, x2, y2 = detected_target
        x_camera, y_camera = int((x1 + x2) / 2), int((y1 + y2) / 2)
        print(f"找到目标 {target_label}，中间点坐标：({x_camera}, {y_camera})")

        calibrator = HandInEyeCalibration()
        robot_x, robot_y = calibrator.get_points_robot(x_camera, y_camera)
        print(f"机械臂坐标 (x, y): {robot_x}, {robot_y}")

        # 执行抓取动作
        robot.MovJ(*home_pose)
        time.sleep(2)

        robot.MovJ(int(robot_x) - 10, int(robot_y), height_pose[2] + 100, *height_pose[3:])
        time.sleep(2)

        robot.MovJ(int(robot_x) - 10, int(robot_y), height_pose[2], *height_pose[3:])
        time.sleep(2)

        robot_init.ToolDOExecute(2, 0)  # 吸取
        time.sleep(2)

        robot.MovJ(int(robot_x) - 10, int(robot_y), height_pose[2] + 100, *height_pose[3:])
        time.sleep(2)

        robot.MovJ(*place_position, *height_pose[3:])
        time.sleep(2.5)

        robot_init.ToolDOExecute(2, 1)  # 松开
        time.sleep(2)

        robot.MovJ(*home_pose)
        print(f"{target_label} 抓取完成，等待下一次指令...\n")
    else:
        print(f"未找到目标 {target_label}，结束本次循环，等待下一次指令...\n")

    time.sleep(2)

capture.release()
cv2.destroyAllWindows()
