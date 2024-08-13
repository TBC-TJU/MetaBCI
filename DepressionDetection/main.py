# -*- coding: utf-8 -*-
#
# Authors: Swolf <Chenxx@emails.bjut.edu.cn>
# Date: 2024/8/01
# License: GNU General Public License v2.0

import numpy as np
from dataset import MetaBCIData
from preprocessing import preprocess_data
from feature_extraction import extract_features
from classification import train_classifier, evaluate_classifier, predict_with_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont

# 设置字体路径
font_path = "E:\BaiduNetdiskDownload\Meta\DepressionDetection.MetaBCI\DepressionDetection\SimHei.ttf"
font_prop = FontProperties(fname=font_path)

labels = ['情绪稳定', '轻微起伏', '中度起伏', '严重起伏']
colors = ['#00FF00', '#FFD700', '#FF4500', '#FF0000']
thresholds = [0, 0.25, 0.5, 0.75, 1.0]

evaluation_texts = {
    '情绪稳定': "您的评估结果为：情绪稳定。这意味着您目前的情绪状态良好，未出现明显的情绪问题。继续保持良好的生活习惯和积极的心态，保持健康的身体和心理状态。",
    '轻微起伏': "您的评估结果为：轻微起伏。轻微起伏可能会导致：出现一些轻微的情绪问题，偶尔感到兴趣丧失或愉快感减少，如焦虑倾向、抑郁倾向，容易出现考前、赛前综合症等。常感觉口苦无味，食欲相对较差，对油腻食物厌恶但喜欢添加刺激性调料。眼睛疲劳、哈欠不断、打盹不止，睡前胡思乱想，可能会伴有入睡困难或易醒、多梦等情况。",
    '中度起伏': "您的评估结果为：中度起伏。中度起伏可能会导致：情绪波动较大，容易产生负面情绪，感到持续的疲劳和乏力。可能会有显著的食欲变化和睡眠问题，如失眠或过度睡眠。需要关注自身的情绪变化，并考虑寻求心理支持。",
    '严重起伏': "您的评估结果为：严重起伏。严重起伏可能会导致：显著的情绪问题，持续的低落情绪，难以体验到快乐。可能伴有严重的焦虑和抑郁症状，出现强烈的无助感和绝望感。建议及时寻求专业的心理帮助，以免情绪问题进一步恶化。"
}

def plot_score_with_text(score, evaluation_text, label):
    # 选择评分对应的颜色和标签
    for i in range(len(thresholds)-1):
        if thresholds[i] <= score < thresholds[i+1]:
            color = colors[i]
            break
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.pie([score, 1-score], colors=[color, '#D3D3D3'], startangle=90, counterclock=True, wedgeprops=dict(width=0.25))
    ax.text(0, 0, f'{label}\n{score:.2f}', ha='center', va='center', fontsize=18, weight='bold', fontproperties=font_prop)
    plt.title('分析结果', fontproperties=font_prop, fontsize=25, pad=30)

    # 添加图例
    legend_labels = [f'{labels[i]} {thresholds[i]:.2f}-{thresholds[i+1]:.2f}' for i in range(len(labels))]
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i], markersize=10, markerfacecolor=colors[i]) for i in range(len(labels))]
    plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(labels), frameon=False, prop=font_prop)

    plt.savefig('temp_plot.png')
    plt.close(fig)

    # 在图像上添加文本
    img = Image.open('temp_plot.png')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 16)
    
    # 将长文本分成多行
    lines = evaluation_text.split('。')
    y_text = img.size[1] - 150
    for line in lines:
        draw.text((15, y_text), line, (0, 0, 0), font=font)
        y_text += 25

    img.save('E:/BaiduNetdiskDownload/Meta/DepressionDetection.MetaBCI/DepressionDetection/output_plot.png')
    img.show()

# 读取健康人和抑郁症患者的数据
healthy_subjects = [f'H{i}' for i in range(1, 17)]
depressed_subjects = [f'D{i}' for i in range(1, 17)]
all_subjects = healthy_subjects + depressed_subjects

# 初始化数据集
dataset = MetaBCIData(subjects=all_subjects, srate=256, paradigm='resting_state')

# 提取特征
X, y = [], []
for subject in healthy_subjects:
    data = dataset.get_data([subject])
    raw = data[subject]['session_1']['run_1']
    raw = preprocess_data(raw)
    psds, freqs, power = extract_features(raw)
    X.append(psds)
    y.append(0)  # 0 表示健康人

for subject in depressed_subjects:
    data = dataset.get_data([subject])
    raw = data[subject]['session_1']['run_1']
    raw = preprocess_data(raw)
    psds, freqs, power = extract_features(raw)
    X.append(psds)
    y.append(1)  # 1 表示抑郁症患者

X = np.array(X)
y = np.array(y)

# 打印调试信息
print(f"X shape: {X.shape}")
if len(X.shape) < 3:
    raise ValueError(f"Unexpected shape for X: {X.shape}. Ensure that extract_features returns features correctly.")

# 获取特征的形状
num_channels, num_features = X.shape[1], X.shape[2]

# 训练分类器
clf = train_classifier(X, y, num_channels=num_channels, num_features=num_features)

# 评估分类器
accuracy, precision, recall, f1 = evaluate_classifier(clf, X, y)
print(f"Classifier accuracy: {accuracy * 100:.2f}%,Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# 预测新数据
new_subject = 'x1new'  # 新数据编号
all_subjects.append(new_subject)  # 将新数据添加到subjects列表中
dataset = MetaBCIData(subjects=all_subjects, srate=256, paradigm='resting_state')  # 重新初始化数据集

new_data = dataset.get_data([new_subject])
raw = new_data[new_subject]['session_1']['run_1']
raw = preprocess_data(raw)
psds, freqs, power = extract_features(raw)

# 打印调试信息
print(f"psds shape: {psds.shape}")

prediction, scores = predict_with_score(clf, np.expand_dims(psds, axis=0))
score = scores[0]
print(f"Prediction for new subject {new_subject}: {'Healthy' if prediction[0] == 0 else 'Depressed'}")
print(f"Depression score for new subject {new_subject}: {score:.2f}")

# 输出抑郁程度
for i in range(len(thresholds)-1):
    if thresholds[i] <= score < thresholds[i+1]:
        label = labels[i]
        evaluation_text = evaluation_texts[label]
        print(f"抑郁程度: {label}")

# 绘制打分图表并添加评估文本
plot_score_with_text(score, evaluation_text, label)



