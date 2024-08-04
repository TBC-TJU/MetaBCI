import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import time
def setup_completed():
    setup_button.config(text="Experiment Setup ✓", background="green")
    experiment_button.config(state=tk.NORMAL)

def start_experiment():
    experiment_button.config(state=tk.DISABLED)
    # 这里可以添加实验的具体内容
    root.withdraw()  # 隐藏窗口
    # 模拟训练过程
    root.deiconify()  # 显示窗口
    for i in range(5):
        progress_var.set((i + 1) * 20)
        accuracy_var.set(f"Accuracy: {i * 20}%")
        root.update()
        time.sleep(2)
        # root.after(1000)  # 模拟训练过程，每隔一秒更新一次


# 创建主窗口
root = tk.Tk()
root.title("Training your model")

# 创建顶部标题
title_label = tk.Label(root, text="Training your model", font=("Helvetica", 16))
title_label.pack(pady=10)

# 创建实验设置部分
setup_button = tk.Button(root, text="Experiment Setup", bg="white", command=lambda: [messagebox.showinfo("Experiment Setup", "Experiment setup completed."), setup_completed()])
setup_button.pack(pady=5)

# 创建实验部分
experiment_button = tk.Button(root, text="Experiment", command=start_experiment, state=tk.DISABLED)
experiment_button.pack(pady=5)

# 创建训练部分
training_label = tk.Label(root, text="Training", font=("Helvetica", 14))
training_label.pack(pady=10)

# 创建进度条和准确率显示
progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(root, length=200, mode='determinate', variable=progress_var)
progress_bar.pack(pady=5)

accuracy_var = tk.StringVar()
accuracy_label = tk.Label(root, textvariable=accuracy_var)
accuracy_label.pack(pady=5)

root.mainloop()