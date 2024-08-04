import numpy as np
import matplotlib.pyplot as plt

# 定义猪腿相关参数
num_bones = 5  # 骨头数量
num_joints = 5  # 关节数量

# 定义关节角度范围（弧度）
joint_angle_ranges = [
    (-np.pi/2, -np.pi/2),   # 第一个关节角度范围
    (-np.pi, -np.pi*3/4),   # 第一个关节角度范围
    (-np.pi*3/4, -np.pi/4),   # 第二个关节角度范围
    (-np.pi, -np.pi*3/4,),   # 第三个关节角度范围
    (-np.pi, -np.pi*1/4)    # 第四个关节角度范围
]

# 定义关节长度
joint_lengths = [1.8, 1, 0.7, 0.4, 0.4]

# 定义猪腿初始位置（关节角度）
initial_joint_angles = [0.0] * num_joints

# 定义重力加速度
gravity = 9.8

# 定义猪腿关节角度变化函数
def update_joint_angles(joint_angles, angle_changes):
    new_joint_angles = []
    for i in range(num_joints):
        new_angle = joint_angles[i] + angle_changes[i]
        new_angle = np.clip(new_angle, joint_angle_ranges[i][0], joint_angle_ranges[i][1])
        new_joint_angles.append(new_angle)
    return new_joint_angles

# 定义猪腿绘制函数
def plot_pig_leg(joint_angles):
    x = [0.0]
    y = [0.0]

    for i in range(num_joints):
        joint_angle = joint_angles[i]
        x.append(x[-1] + joint_lengths[i] * np.cos(joint_angle))
        y.append(y[-1] + joint_lengths[i] * np.sin(joint_angle))

    plt.plot(x, y, 'bo-')
    plt.axis('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Pig Leg Motion')

# 模拟猪腿运动
num_steps = 100  # 模拟步数
angle_changes = np.random.uniform(-0.5,0.5, size=(num_steps, num_joints))
joint_angles = initial_joint_angles

plt.figure(figsize=(8, 8))
for i in range(num_steps):
    plt.clf()
    joint_angles = update_joint_angles(joint_angles, angle_changes[i])
    angle_changes[i][0] -= gravity * np.sin(joint_angles[0])  # 考虑重力效果
    plot_pig_leg(joint_angles)
    plt.pause(0.05)

plt.show()