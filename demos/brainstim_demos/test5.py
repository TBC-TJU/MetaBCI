'''
test5
用于测试键盘、鼠标监听，其他线程指令监听，根据指令操作鼠标、键盘
'''
from pynput.keyboard import Key, Listener
from pynput.mouse import Button, Controller

# 此时的Listener是从keyboard里面导入的
mouse = Controller()

def on_press(key):
    # 当按下esc，结束监听
    if key == Key.esc:
        print(f"你按下了esc，监听结束")
        return False
    print(f"你按下了{key}键")
    if key == Key.up:
        mouse_x, mouse_y = mouse.position
        mouse_y -= 10
        mouse.position = (mouse_x, mouse_y)


def on_release(key):
    print(f"你松开了{key}键")


with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()