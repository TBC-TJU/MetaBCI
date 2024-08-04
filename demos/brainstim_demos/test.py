import pyglet
from pyglet.window import mouse

# 创建窗口
window = pyglet.window.Window(width=800, height=600, )

# 创建方块
square = pyglet.shapes.Rectangle(x=100, y=100, width=100, height=100)

# 设置鼠标状态变量
is_dragging = False
offset_x = 0
offset_y = 0

@window.event
def on_draw():
    window.clear()
    square.draw()

@window.event
def on_mouse_press(x, y, button, modifiers):
    global is_dragging, offset_x, offset_y
    if button == mouse.LEFT and square.x < x < square.x + square.width and square.y < y < square.y + square.height:
        is_dragging = True
        offset_x = square.x - x
        offset_y = square.y - y

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if is_dragging:
        square.x = x + offset_x
        square.y = y + offset_y

@window.event
def on_mouse_release(x, y, button, modifiers):
    global is_dragging
    if button == mouse.LEFT:
        is_dragging = False

pyglet.app.run()