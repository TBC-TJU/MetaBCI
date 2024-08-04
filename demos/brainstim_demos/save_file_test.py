from itertools import zip_longest
import datetime
import os
import tkinter as tk

class save_flie():
    def __init__(self, root_title="save_file"):
        self.root = tk.Tk()
        #self.root.overrideredirect(True)
        self.root.attributes("-alpha", 0.85)
        self.root.title(root_title)
        self.dict_judge = dict()
        self.dict_judge["identity_judge"] = None
        self.dict_judge["file_name"] = None
        self.text_read_1 = "Editor please input 'e' OR User, please input 'u'"
        self.text_read_2 = "Name of the new stim:"
        self.text_read_3 = ("Name of the new stim:")
        self.entry_label_main = tk.Label(self.root, text=self.text_read_1, font=('Times New Roman', 25))
        self.entry_filename_main = tk.Entry(self.root, font=('Times New Roman', 25), width=10, justify=tk.CENTER,
                                       state=tk.NORMAL)
        self.entry3 = tk.Label(self.root, text=self.text_read_2, font=('Times New Roman', 25))
        self.entry4 = tk.Entry(self.root, font=('Times New Roman', 25), width=30, justify=tk.CENTER, state=tk.NORMAL)
        self.entry5 = tk.Label(self.root, text=self.text_read_3, font=('Times New Roman', 25))
        self.entry6 = tk.Entry(self.root, font=('Times New Roman', 25), width=30, justify=tk.CENTER, state=tk.NORMAL)

        self.tkinter_creat()

        self.root.mainloop()


    def tkinter_creat(self):
        self.entry_label_main.pack()
        self.entry_filename_main.pack()
        self.entry_filename_main.bind("<Return>", self.on_entry_return)
        choice = self.entry_filename_main.get()
        return choice

    def filename_return(self, event, entry_widget):
        file_name = entry_widget.get()
        self.dict_judge["file_name"] = file_name
        # 关闭窗口
        self.root.destroy()

    def on_entry_return(self, event):
        get_choice = self.tkinter_creat()
        if get_choice == 'e':
            self.dict_judge["identity_judge"] = "editor"
            self.entry3.pack()
            self.entry4.pack()
            # 为新 entry 绑定 Return 键
            self.entry4.bind("<Return>", lambda e: self.filename_return(e, self.entry4))
        elif get_choice == 'u':
            self.dict_judge["identity_judge"] = "user"
            self.entry5.pack()
            self.entry6.pack()
            # 为新 entry 绑定 Return 键
            self.entry6.bind("<Return>", lambda u: self.filename_return(u, self.entry6))
        else:
            print('invalid input')


    def save(self, ssvep_keys1):

        # l1 = ['paradigm', 'name', 'stim_names', 'stim_pos', 'n_elements',
        #       'stm_length', 'stm_width','stm_time','freqs','phases','key_mouse_mapping']
        # l2 = ssvep_keys1
        # zipped = zip_longest(l1, l2, fillvalue=None)
        # dict_save = {k: v for k, v in zipped if k is not None}  # 排除键为 None 的情况
        # print(dict_save)

        user_home = os.path.expanduser('~')
        # 创建一个名为 'stim_position' 的新目录路径
        user_dir = os.path.join(user_home, 'AssistBCI\\stim_position')
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)



        if self.dict_judge["identity_judge"] == "editor":
            file_name = "#" + self.dict_judge.get("file_name", "") + ".txt"
            file_path = os.path.join(user_dir, file_name)
            with open(file_path, "w") as file:  # 使用 "w" 模式，因为我们从头开始写
                # 遍历 dit 中的每个条目
                file.write(f'name="{"#" + self.dict_judge.get("file_name", "")}"\n')
                for key in ssvep_keys1.keys():
                    file.write(f'{key}={ssvep_keys1[key]}\n')

        elif self.dict_judge["identity_judge"] == "user":

            if self.dict_judge["file_name"] == "":
                existing_files = os.listdir(user_dir)
                # 计算与“configure_[i].txt”模式相匹配的文件数量
                count = sum(1 for f in existing_files if f.startswith("configure_") and f.endswith(".txt"))

                filename = f"configure_{count}.txt"
                file_path = os.path.join(user_dir, filename)
                with open(file_path, "w") as file:  # 使用 "w" 模式，因为我们从头开始写
                    # 遍历 dit 中的每个条目
                    file.write(f'name="configure_{count}"\n')
                    for key in ssvep_keys1.keys():
                        # 写入宽度和高度
                        file.write(f'{key}={ssvep_keys1[key]}\n')

            else:
                file_name =self.dict_judge.get("file_name", "") + ".txt"
                file_path = os.path.join(user_dir, file_name)
                with open(file_path, "w") as file:  # 使用 "w" 模式，因为我们从头开始写
                    file.write(f'name="{self.dict_judge.get("file_name", "")}"\n')
                    # 遍历 dit 中的每个条目
                    for key in ssvep_keys1.keys():
                        # 写入宽度和高度
                        file.write(f'{key}={ssvep_keys1[key]}\n')

        else:
            self.dict_judge["identity_judge"] = None
            print('error happened')