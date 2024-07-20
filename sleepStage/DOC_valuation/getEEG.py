import sys
sys.path.append('.')
sys.path.append('src/main\java/com/nj/back/MCASleepNet_sys/')
import argparse
import os
import numpy as np
#用于系统的脑电监测，读取da
import pyedflib
import pandas as pd

np.set_printoptions(threshold=np.inf)

#用于系统的脑电监测，读取data的edf和预处理edf后生成的npz
def getData(file,select=0,begin=0):
    # file = "E:\IDEA\MCA\excel\patient_mcs_modehuai_20180928_1.xls"
    format=file.split(".")[1]
    if format == 'xlsx':
        format='xls'
    if format == 'npz': #npz
        # npz#系统的脑电监测，读取npz文件
        # file = "liangxiaotong_mcs_20181018_1_00.npz"
        # file_path = "E:\sleepModel\MCASleepNet\data\sleepedf\sleep-cassette\eeg_eog" + "/" + file
        # 加载.npz文件
        data = np.load(file)
        # print(data.files)
        # 创建一个空的DataFrame
        columns = ['x', 'y']
        data2 = []
        for key in columns:
            if key == 'fs':
                break
            data2.append(data[str(key)])
        # print(data2)
        df = pd.DataFrame(columns=columns)
        # print(df)
        # 将.npz文件中的每个数组添加到DataFrame中
        # print(data2[2])
        data2[0] = np.ravel(data2[0])
        data2[1] = np.ravel(data2[1])
        # df['x'] = data2[0]
        # df['y'] = data2[1]
        length = len(data2[select])
        if (begin > length):
            begin = length-500
        if begin <0:
            begin = 0
        n = begin + 500
        if n>length:
            n=length
        result = ""
        for i in np.arange(begin,n):
            if i == n - 1:
                result += ("%.3f" % data2[select][i])
            else:
                result += ("%.3f," % data2[select][i])
        print(result)
    elif format == 'edf': # edf
        #系统的脑电监测，读取edf文件
        f = pyedflib.EdfReader(file)
        channel = 0
        for i in range(len(f.getSignalLabels())):
            if select==0 and f.getSignalLabels()[i] == 'C3':
                channel = i
                break
            elif select==1 and 'EOG' in f.getSignalLabels()[i]:
                channel =i
                break
            else:
                channel=select
        buf = f.readSignal(channel)
        length = len(buf)
        if (begin > length):
            begin = length - 500
        if begin <0:
            begin = 0
        n = begin + 500
        if n > length:
            n = length
        result = ""
        for i in np.arange(begin,n):
            if i ==n-1:
                result += ("%.3f" % buf[i])
            else:
                result += ("%.3f," % buf[i])
        print(result)
        f.close()
        del f
    elif format == 'xls': #excel
        data=pd.read_excel(file)
        columns = ['x', 'y']
        length = len(data[columns[select]])
        if begin > length:
            begin = length - 500
        if begin <0:
            begin = 0
        n = begin + 500
        if n > length:
            n = length
        result = ""
        for i in np.arange(begin, n):
            if i == n - 1:
                result += ("%.3f" % data[columns[select]][i])
            else:
                result += ("%.3f," % data[columns[select]][i])
        print(result)
    elif format == 'csv': #csv
        data = pd.read_csv(file)
        columns = ['x', 'y']
        length = len(data[columns[select]])
        if begin > length:
            begin = length - 500
        if begin <0:
            begin = 0
        n = begin + 500
        if n > length:
            n = length
        result = ""
        for i in np.arange(begin, n):
            if i == n - 1:
                result += ("%.3f" % data[columns[select]][i])
            else:
                result += ("%.3f," % data[columns[select]][i])
        print(result)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=False)
    parser.add_argument("--select", type=int, required=False,default=0)
    parser.add_argument("--begin", type=int, required=False, default=0)
    args = parser.parse_args()

    getData(
        file=args.file,
        select=args.select,
        begin = args.begin,
    )
