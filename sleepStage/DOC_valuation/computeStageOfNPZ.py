import sys
import argparse
import os
import numpy as np
import pandas as pd
import pyedflib
np.set_printoptions(threshold=np.inf)

# Label values
W = 0       # Stage AWAKE
N1 = 1      # Stage N1
N2 = 2      # Stage N2
N3 = 3      # Stage N3
REM = 4     # Stage REM
MOVE = 5    # Movement
UNK = 6     # Unknown

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK,
}
class_dict = {
    W: "W",
    N1: "N1",
    N2: "N2",
    N3: "N3",
    REM: "REM",
    MOVE: "MOVE",
    UNK: "UNK",
}

# 读取预测predict后的npz
def stageOfNpz(file):
    data = np.load(file,allow_pickle=True)
    length=len(data['z_pred'])
    list4 = [] #最后返回的字符串数组
    list4.append(str(data['start_datetime']))
    list4.append(str("%.2fh" % (data['file_duration']/3600)))
    time=length/120 #睡眠时长 小时
    list4.append(("%.2f" % time)+"h") #总睡眠时长 小时
    list = [0,0,0,0,0] #对应w,(n1,n2),n3,rem
    for i in data['z_pred']:
        list[int(i)]+=1
    list2 = [] #浮点数数组
    # list3 = [] #字符串数值
    for i in list:
        temp = (i / length * time)
        list2.append(temp)
        list4.append(("%.2f" % temp)+"h") #对应w,n1,n2,n3,rem的睡眠时长 小时
    for i in list2:
        ss = str("%.2f" % (i / time * 100)) + "%"  # 深睡比例
        list4.append(ss)
    result = ""
    n = len(list4)
    for i in np.arange(0, n):
        if i == n - 1:
            result += list4[i]
            break
        else:
            result += list4[i]+','
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    stageOfNpz(
        file=args.file,
    )
