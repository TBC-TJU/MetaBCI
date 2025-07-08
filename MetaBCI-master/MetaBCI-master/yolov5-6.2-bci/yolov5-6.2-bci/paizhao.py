import cv2
import datetime

# 打开默认摄像头（索引为 0），Windows 用户推荐使用 CAP_DSHOW
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✅ 摄像头已开启")
print("📸 按空格键拍照，按 q 退出")

while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取帧")
        break

    # 显示摄像头画面
    cv2.imshow("摄像头 - 按空格拍照", frame)

    key = cv2.waitKey(1) & 0xFF

    # 按空格键拍照
    if key == 32:  # 空格键
        # 保存图像
        filename = datetime.datetime.now().strftime("1.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ 拍照成功，已保存为 {filename}")

    # 按 'q' 键退出
    elif key == ord('q'):
        print("👋 退出程序")
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
