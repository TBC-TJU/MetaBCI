import pyglet
from pyglet.window import mouse


def set_stim_beta(buffer, *args, **kwargs):

    # 创建窗口
    window = pyglet.window.Window(*args, **kwargs)

    # # 创建方块
    # square = pyglet.shapes.Rectangle(x=100, y=100, width=100, height=100)

    squares = []

    # 创建按钮
    button_bigger = pyglet.shapes.Rectangle(x=100, y=100, width=100, height=100)
    button_smaller = pyglet.shapes.Rectangle(x=250, y=100, width=100, height=100)
    button_add = pyglet.shapes.Rectangle(x=400, y=100, width=100, height=100)
    button_save = pyglet.shapes.Rectangle(x=550, y=100, width=100, height=100)
    button_bigger.color = (127, 127, 127)
    button_smaller.color = (127, 127, 127)
    button_add.color = (127, 127, 127)
    button_save.color = (127, 127, 127)

    global key_flag
    key_flag = []

    offset_x = 0
    offset_y = 0

    @window.event
    def is_button_clicked(button, x, y):
        return button.x < x < button.x + button.width and button.y < y < button.y + button.height


    @window.event
    def on_draw():
        window.clear()
        for square in squares:
            square.draw()
        button_bigger.draw()
        button_smaller.draw()
        button_add.draw()
        button_save.draw()

    @window.event
    def on_mouse_press(x, y, input, modifiers):
        global key_flag, offset_x, offset_y, buffer
        if input == mouse.LEFT:
            if is_button_clicked(button_bigger, x, y):
                for i, square in enumerate(squares):
                    if key_flag[i]:
                        button_bigger.color = (0, 255, 0)  # 设置按钮颜色为绿色
                        square.width += 10
                        square.height += 10

            elif is_button_clicked(button_smaller, x, y):
                for i, square in enumerate(squares):
                    if key_flag[i]:
                        button_smaller.color = (0, 255, 0)  # 设置按钮颜色为绿色
                        square.width -= 10
                        square.height -= 10

            elif is_button_clicked(button_add, x, y):
                button_add.color = (0, 255, 0)  # 设置按钮颜色为绿色色
                squares.append(pyglet.shapes.Rectangle(x=100, y=100, width=200, height=200, color=(0, 255, 0)))

                for m in range(len(key_flag)):
                    key_flag[m] = False
                key_flag.append(True)


            elif is_button_clicked(button_save, x, y):
                button_save.color = (0, 255, 0)
                for i, square in enumerate(squares):
                    buffer.append([square.width, square.height, square.color, square.x, square.y])
                window.close()


            else:
                for i, square in enumerate(squares):
                    if is_button_clicked(square, x, y):

                        for m in range(len(key_flag)):
                            key_flag[m] = False

                        if True not in key_flag:
                            key_flag[i] = True
                            square.color = (0, 255, 0)
                            offset_x = square.x - x
                            offset_y = square.y - y
                        else:
                            key_flag[i] = False
                    else:
                        key_flag[i] = False



    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        global key_flag, offset_x, offset_y
        for i, square in enumerate(squares):
            if key_flag[i]:
                square.x = x + offset_x
                square.y = y + offset_y

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        global key_flag
        if button == mouse.LEFT:
            for i, square in enumerate(squares):
                if not key_flag[i]:
                    square.color = (127, 127, 127)

            if is_button_clicked(button_bigger, x, y):
                button_bigger.color = (127, 127, 127)

            if is_button_clicked(button_smaller, x, y):
                button_smaller.color = (127, 127, 127)

            if is_button_clicked(button_add, x, y):
                button_add.color = (127, 127, 127)

    pyglet.app.run()



buffer = []
set_stim_beta(buffer, width=800, height=600, style='transparent')
print(buffer)