import pyglet
from pyglet.window import key
from pynput.mouse import Button, Controller

# 创建一个透明窗口
window = pyglet.window.Window(width=1920, height=1080, style=pyglet.window.Window.WINDOW_STYLE_OVERLAY)
window.set_location(100, 100)
window.set_mouse_visible(True)
window.set_exclusive_mouse(False)

# 定义鼠标的初始位置
mouse_x = window.width // 2
mouse_y = window.height // 2
mouse = Controller()

@window.event
def on_draw():
    window.clear()

@window.event
def on_key_press(symbol, modifiers):
    global mouse_x, mouse_y

    # 按下上键，向上移动鼠标
    if symbol == key.UP:
        mouse_x, mouse_y = mouse.position
        mouse_y -= 10

    # 按下下键，向下移动鼠标
    elif symbol == key.DOWN:
        mouse_x, mouse_y = mouse.position
        mouse_y += 10

    # 按下左键，向左移动鼠标
    elif symbol == key.LEFT:
        mouse_x, mouse_y = mouse.position
        mouse_x -= 10

    # 按下右键，向右移动鼠标
    elif symbol == key.RIGHT:
        mouse_x, mouse_y = mouse.position
        mouse_x += 10

    elif symbol == key.ENTER:
        mouse_x, mouse_y = mouse.position
        mouse_x += 10

    # 更新窗口的相对位置来模拟移动鼠标
    mouse.position = (mouse_x, mouse_y)

pyglet.app.run()