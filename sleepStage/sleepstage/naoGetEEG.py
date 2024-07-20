import argparse
import os
import numpy as np
import pyedflib
import pandas as pd

def getEEg(select=0,begin=0):
    j=begin
    aa = []
    path = os.environ.get("SLEEPNET_JAR_PATH")
    file = open(path + '//' + 'dd.txt', 'r')
    s = file.read()
    o = s.split()
    bb = o[j:j + 15000]
    # print("bb的长度")
    # print(len(bb))
    file.close()
    aa = list(int(i) for i in bb)
    # aa = list(int(i) for i in o)
    bb = aa
    # print(len(aa))

    X1 = bb
    X2 = aa
    data2 = []
    data2.append(X1)
    data2.append(X2)
    length = len(data2[select])
    if (begin > length):
        begin = length - 500
    if begin < 0:
        begin = 0
    n = begin + 500
    if n > length:
        n = length
    result = ""
    for i in np.arange(begin, n):
        if i == n - 1:
            result += ("%.3f" % data2[select][i])
        else:
            result += ("%.3f," % data2[select][i])
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", type=int, required=False,default=0)
    parser.add_argument("--begin", type=int, required=False, default=0)
    args = parser.parse_args()

    getEEg(
        select=args.select,
        begin = args.begin,
    )