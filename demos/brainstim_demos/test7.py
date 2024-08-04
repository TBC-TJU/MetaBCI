import time
from demos.brainstim_demos.key_mouse_beta import Virtual_Output


CMD_list = [
    'mouse up 100',
    'mouse down 100',
    'mouse left 200',
    'mouse right 200',
    'mouse button_left 1',
    'mouse button_right 1',
    'mouse left 50',
    'mouse button_left 1',
    'mouse scroll_up 50',
    'mouse scroll_down 50'
]

if __name__ == '__main__':
    a = Virtual_Output()
    a.start()
    for i in range(100):
        for CMD in CMD_list:
            time.sleep(0.5)
            print("CMD: ", CMD)
            a.CMD(CMD)
    a.stop()