import sys
import argparse
import os
import numpy as np
import pandas as pd
import pyedflib
np.set_printoptions(threshold=np.inf)
import datetime

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
    # W: "W",
    # N1: "N1",
    # N2: "N2",
    # N3: "N3",
    # REM: "REM",
    # MOVE: "MOVE",
    # UNK: "UNK",
    W: "W期",
    N1: "N1期",
    N2: "N2期",
    N3: "N3期",
    REM: "REM期",
    MOVE: "Movement",
    UNK: "未知",
}

# 读取预测predict后的npz
def stageOfNpz(file,begin=0):
    data = np.load(file,allow_pickle=True)
    startTime = str(data['start_datetime'])
    time = datetime.datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S")
    # 计算偏移量
    offset = datetime.timedelta(seconds=+30)
    # 获取修改后的时间并格式化
    length=len(data['z_pred'])
    if begin > length:
        begin = length-500
    if begin < 0:
        begin = 0
    n = begin + 100000
    if n>length:
        n=length
    result = ""
    for i in np.arange(begin,n):
        if i == n - 1:
            s = str((time + offset).strftime("%Y-%m-%d %H:%M:%S"))
            result += ("%s" % s)+" "+("%s" % class_dict[data['z_pred'][i]])
            break
        else:
            s = str((time + offset).strftime("%Y-%m-%d %H:%M:%S"))
            result += ("%s" % s)+" "+("%s," % class_dict[data['z_pred'][i]])
            time = (time + offset)
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--begin", type=int, required=False, default=0)
    args = parser.parse_args()

    stageOfNpz(
        file=args.file,
        begin=args.begin,
    )
