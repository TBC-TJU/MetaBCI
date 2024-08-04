import pyglet
import ctypes

ctypes.windll.user32.SetProcessDPIAware()

# 定义界面相关参数
window_width = 3120
window_height = 2079

selected_line = 0

menu_layer = [0, ]
menu_element = [0, ]

# 创建窗口
window = pyglet.window.Window(window_width, window_height, style='overlay')

# 创建标签列表
title_list = {}


titles = ["General Mode", "Gaming Mode", "Stim KeyBoard Setup", "Stim KeyBoard Management", "Train Personal Model", "More APP..."]
labels = []
line_count = len(titles)
line_height = window_height / (line_count + 4)
for i in range(line_count):
    label = pyglet.text.Label(
        titles[i],
        font_name='Arial',
        font_size=35,
        x=window_width/6,
        y=window_height - (i + 2) * line_height,
        anchor_x='left',
        anchor_y='baseline',
        color=(255, 255, 255, 255)  # 初始颜色为白色
    )
    labels.append(label)
title_list[str(0)+str(0)] = labels


labels = []
titles = ["Glaucoma Detection", "SSVEP Picture Book", "More APP..."]
line_count = len(titles)
line_height = window_height / (line_count + 4)
for i in range(line_count):
    label2 = pyglet.text.Label(
        titles[i],
        font_name='Arial',
        font_size=35,
        x=window_width/6,
        y=window_height - (i + 2) * line_height,
        anchor_x='left',
        anchor_y='baseline',
        color=(255, 255, 255, 255)  # 初始颜色为白色
    )
    labels.append(label2)
title_list[str(1)+str(5)] = labels


labels = []
titles = ["More APP..."]
line_count = len(titles)
line_height = window_height / (line_count + 4)
for i in range(line_count):
    label2 = pyglet.text.Label(
        titles[i],
        font_name='Arial',
        font_size=35,
        x=window_width/6,
        y=window_height - (i + 2) * line_height,
        anchor_x='left',
        anchor_y='baseline',
        color=(255, 255, 255, 255)  # 初始颜色为白色
    )
    labels.append(label2)
title_list[str(2)+str(2)] = labels


@window.event
def on_draw():
    global menu_layer, menu_element
    window.clear()
    for i in range(len(title_list[str(menu_layer[-1])+str(menu_element[-1])])):
        title_list[str(menu_layer[-1])+str(menu_element[-1])][i].color = (255, 165, 0, 255) if i == selected_line else (255, 255, 255, 255)
        title_list[str(menu_layer[-1])+str(menu_element[-1])][i].draw()


@window.event
def on_key_press(symbol, modifiers):
    global selected_line, menu_layer, menu_element
    if symbol == pyglet.window.key.UP:
        selected_line = max(0, selected_line - 1)
    elif symbol == pyglet.window.key.DOWN:
        selected_line = min(len(title_list[str(menu_layer[-1])+str(menu_element[-1])]) - 1, selected_line + 1)
    elif symbol == pyglet.window.key.ENTER:
        if str(menu_layer[-1]+1)+str(selected_line) in title_list:
            menu_element.append(selected_line)
            menu_layer.append(menu_layer[-1]+1)
            selected_line = 0
            print("elements:", menu_element)
            print("layers:", menu_layer)
    elif symbol == pyglet.window.key.LEFT:
        if len(menu_element)>1 and len(menu_layer)>1:
            selected_line = menu_element[-1]
            menu_element = menu_element[:-1]
            menu_layer = menu_layer[:-1]





pyglet.app.run()