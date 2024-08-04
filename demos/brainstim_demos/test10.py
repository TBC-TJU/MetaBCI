from pynput import keyboard
control = keyboard.Controller()
control.press(keyboard.Key.cmd)
control.release(keyboard.Key.cmd)