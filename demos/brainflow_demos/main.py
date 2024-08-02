# main.py
from matdataset import MetaBCIData

# 示例如何使用这个类
dataset = MetaBCIData(
    subjects=list(range(1, 14)),  # 受试者编号从 1 到 13
    srate=256,  # 示例采样率，实际需根据数据设置
    paradigm='resting_state'  # 示例范式
)

# 获取受试者 1 的数据
data = dataset.get_data([1])
print(data)

