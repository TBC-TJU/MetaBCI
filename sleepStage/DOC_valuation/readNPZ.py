import sys
import argparse
import os
import numpy as np
import pandas as pd
import pyedflib
np.set_printoptions(threshold=np.inf)

# 读取预测predict后的npz
def readNpz(file,select=0,begin=0):
    # file="E:\sleepModel\MCASleepNet_sys\out_sleepedf\predict\pred_patient_mcs_modehuai_20180928_1.npz"
    if select == 0: # 分类
        # npz#系统的脑电监测，读取npz文件
        # file = "liangxiaotong_mcs_20181018_1_00.npz"
        # file_path = "E:\sleepModel\MCASleepNet\data\sleepedf\sleep-cassette\eeg_eog" + "/" + file
        # 加载.npz文件
        data = np.load(file)
        flag=data['pred']
        if flag > 0.5:
            print("1")#MCS
        else:
            print("0")#VS
    elif select == 1: # 分期
        data = np.load(file)
        # print("read %i samples" % n)
        length=len(data['z_pred'])
        if begin>length:
            begin=length-500
        if begin <0:
            begin = 0
        n = begin + 10000
        if n>length:
            n=length
        result = ""
        for i in np.arange(begin,n):
            if i == n - 1:
                result += ("%.3f" % data['z_pred'][i])
                break
            else:
                result += ("%.3f," % data['z_pred'][i])
        print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--select", type=int, required=True,default=0)
    parser.add_argument("--begin",type=int,required=False,default=0)
    args = parser.parse_args()

    readNpz(
        file=args.file,
        select=args.select,
        begin=args.begin,
    )
