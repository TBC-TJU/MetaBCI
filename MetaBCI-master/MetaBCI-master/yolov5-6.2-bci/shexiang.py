import cv2

# 打开默认摄像头（设备编号为0）
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 逐帧捕获
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        print("无法读取摄像头画面")
        break

    # 显示帧
    cv2.imshow("摄像头画面", frame)

    # 按下 q 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
