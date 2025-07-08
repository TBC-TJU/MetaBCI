import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from imageio import imsave
# import h5py

xmin, ymin, w, h = 120, 90, 400, 300


def depth2Gray(im_depth):
    """
    将深度图转至三通道8位灰度图
    (h, w, 3)
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError

    k = 255 / (x_max - x_min)
    b = 255 - k * x_max

    ret = (im_depth * k + b).astype(np.uint8)
    return ret


def depth2RGB(im_depth):
    """
    将深度图转至三通道8位彩色图
    先将值为0的点去除,然后转换为彩图,然后将值为0的点设为红色
    (h, w, 3)
    im_depth: 单位 mm或m
    """
    im_depth = depth2Gray(im_depth)
    im_color = cv2.applyColorMap(im_depth, cv2.COLORMAP_JET)
    return im_color


def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.在深度图像中补绘缺失值。
    :param missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.缩放保持为浮动，但必须在边界-1:1保持
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.恢复到原来的大小和值范围。
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


def run():
    pipeline = rs.pipeline()

    # Create a config并配置要流​​式传输的管道
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)

    save_path = 'realsense'
    saved_count = 0

    os.makedirs(os.path.join((save_path), "task{:d}".format(saved_count)), exist_ok=True)
    #
    # # 初始化 h5 文件
    # h5 = h5py.File(os.path.join((save_path), "task{:d}".format(saved_count), "task{:d}.hdf5".format(saved_count)),
    #                'w')  # 创建一个空的h5文件

    # 初始化 avi文件
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    avi = cv2.VideoWriter(os.path.join((save_path), "task{:d}".format(saved_count), "task{:d}.avi".format(saved_count)),
                          fourcc, 30, (640, 480), True)

    # 保存视频
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

    pipeline.start(config)
    # 主循环
    try:
        id = 0
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            # 获取RGB图像
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            # color_image = color_image[ymin:ymin + h, xmin:xmin + w].copy()
            # color_image = cv2.resize(color_image, (640, 480))
            # 获取深度图
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32) / 1000.  # 单位为m
            # 可视化图像
            depth_image = inpaint(depth_image)  # 补全缺失值
            depth_image_color = depth2RGB(depth_image)
            # depth_image_color = depth_image_color[ymin:ymin + h, xmin:xmin + w].copy()
            # depth_image_color = cv2.resize(depth_image_color, (640, 480))
            cv2.imshow("video", np.hstack((color_image, depth_image_color)))
            key = cv2.waitKey(30)

            # s 保存图片
            if key & 0xFF == ord('s'):
                cv2.imwrite(os.path.join((save_path), "pcd{:04d}r.png".format(saved_count)), color_image)  # 保存RGB为png文件
                imsave(os.path.join((save_path), "pcd{:04d}d.tiff".format(saved_count)), depth_image)  # 保存深度图为tiff文件
                saved_count+=1
                cv2.imshow("save", np.hstack((color_image, depth_image_color)))

            # 把彩色图写入 avi
            avi.write(color_image)
            # 把深度图写入 h5
            _, depth_map_encode = cv2.imencode('.tiff',
                                               depth_image_color)  # 编码depth_map 将depth_map 传入, res为bool, 用于判断是否编码成功
            # .png的原因: opencv 支持将16位图片写入.png, .jpg仅能写入8位图片
            # h5[str(id)] = depth_map_encode  # h5类似于字典, 即key 对应 data, 并且key不能重复
            id += 1

            # q 退出
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        # h5.close()  # 关闭文件

    finally:
        pipeline.stop()


if __name__ == '__main__':
    run()
