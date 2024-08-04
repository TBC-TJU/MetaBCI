import pyglet
from pyglet.window import key
from pyglet.window import mouse


def set_stim_beta(buffer, *args, **kwargs):

    # 创建窗口
    window = pyglet.window.Window(*args, **kwargs)

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

    global key_states
    key_states = {}

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
    def on_key_press(symbol, modifiers):
        global key_states
        key_states[symbol] = True

        if key_states.get(key.B):
            for i, square in enumerate(squares):
                if key_states.get(key.LSHIFT) and key_states.get(key.LCTRL):
                    button_bigger.color = (0, 255, 0)  # 设置按钮颜色为绿色
                    square.width += 10
                    square.height += 10
        elif key_states.get(key.S):
            for i, square in enumerate(squares):
                if key_states.get(key.LSHIFT) and key_states.get(key.LCTRL):
                    button_smaller.color = (0, 255, 0)  # 设置按钮颜色为绿色
                    square.width -= 10
                    square.height -= 10
        elif key_states.get(key.A):
            button_add.color = (0, 255, 0)  # 设置按钮颜色为绿色色
            squares.append(pyglet.shapes.Rectangle(x=100, y=100, width=200, height=200, color=(0, 255, 0)))

            for m in range(len(key_states)):
                key_states[m] = False
            key_states[len(key_states)] = True
        elif key_states.get(key.F):
            button_save.color = (0, 255, 0)
            for i, square in enumerate(squares):
                buffer.append([square.width, square.height, square.color, square.x, square.y])
            window.close()

    @window.event
    def on_key_release(symbol, modifiers):
        global key_states
        key_states[symbol] = False

    @window.event
    def on_mouse_press(x, y, input, modifiers):
        global key_states, offset_x, offset_y, buffer
        if input == mouse.LEFT:
            if is_button_clicked(button_bigger, x, y):
                for i, square in enumerate(squares):
                    if key_states.get(i):
                        button_bigger.color = (0, 255, 0)  # 设置按钮颜色为绿色
                        square.width += 10
                        square.height += 10

            elif is_button_clicked(button_smaller, x, y):
                for i, square in enumerate(squares):
                    if key_states.get(i):
                        button_smaller.color = (0, 255, 0)  # 设置按钮颜色为绿色
                        square.width -= 10
                        square.height -= 10

            elif is_button_clicked(button_add, x, y):
                button_add.color = (0, 255, 0)  # 设置按钮颜色为绿色色
                squares.append(pyglet.shapes.Rectangle(x=100, y=100, width=200, height=200, color=(0, 255, 0)))

                for m in range(len(key_states)):
                    key_states[m] = False
                key_states[len(key_states)] = True

            elif is_button_clicked(button_save, x, y):
                button_save.color = (0, 255, 0)
                for i, square in enumerate(squares):
                    buffer.append([square.width, square.height, square.color, square.x, square.y])
                window.close()

            else:
                for i, square in enumerate(squares):
                    if is_button_clicked(square, x, y):

                        for m in range(len(key_states)):
                            key_states[m] = False

                        if True not in key_states.values():
                            key_states[i] = True
                            square.color = (0, 255, 0)
                            offset_x = square.x - x
                            offset_y = square.y - y
                        else:
                            key_states[i] = False
                    else:
                        key_states[i] = False

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        global key_states, offset_x, offset_y
        for i, square in enumerate(squares):
            if key_states.get(i):
                square.x = x + offset_x
                square.y = y + offset_y

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        global key_states
        if button == mouse.LEFT:
            for i, square in enumerate(squares):
                if not key_states.get(i):
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