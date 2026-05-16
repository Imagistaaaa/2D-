import numpy as np
import matplotlib.pyplot as plt

#全局参数
theta1_past = theta2_past = 0.0
l1 = 1.0
l2 = 1.0
step = 50
origin = (0, 0)
#----------------------------------


#计算两个关节的坐标
def forward_kinematics(theta1, theta2):
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return origin, (x1, y1), (x2, y2)
#----------------------------------


#插值生成轨迹
def generate_trajectory(theta1_last, theta2_last, theta1_new, theta2_new):
    theta1s = np.linspace(theta1_last, theta1_new, num=step)
    theta2s = np.linspace(theta2_last, theta2_new, num=step)
    trail = []
    for t1, t2 in zip(theta1s, theta2s):
        trail.append(forward_kinematics(t1, t2)[2])
    return trail
#----------------------------------


#修正角度,以避免出现过半圈跳转
def correcct_degree(theta_cur,theta_past):
    theta_cur = np.radians(theta_cur)
    if abs(theta_cur-theta_past)>np.pi:
        if theta_cur>theta_past:
            theta_cur = theta_cur - 2*np.pi
        else:
            theta_cur = theta_cur + 2*np.pi
    return theta_cur
#----------------------------------


#绘制轨迹和机械臂
def plot(theta1_last, theta2_last, theta1_new, theta2_new):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    
    last = np.array(forward_kinematics(theta1_last, theta2_last))
    new  = np.array(forward_kinematics(theta1_new, theta2_new))
    trail = np.array(generate_trajectory(theta1_last, theta2_last,
                                         theta1_new, theta2_new))
    
    ax.plot(last[:,0], last[:,1], color='b', linestyle='-.', linewidth=2,
            marker='o', markersize=4, markerfacecolor='w', markeredgecolor='b',
            label='arms last')
    ax.plot(new[:,0], new[:,1], color='b', linestyle='-', linewidth=6,
            marker='o', markersize=8, markerfacecolor='w', markeredgecolor='b',
            label='arms new')
    ax.plot(trail[:,0], trail[:,1], color='r', linestyle='-', linewidth=2,
            label='trail')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()  
    
#-----------------------------------


#主程序
def main():
    global theta1_past, theta2_past
    while True:
        inp = input('theta1 theta2 (q 退出): ').strip()
        if inp.lower() == 'q':
            break
        parts = inp.split()
        if len(parts) != 2:
            print("请输入两个角度值（度）")
            continue
        theta1_cur = correcct_degree(float(parts[0]), theta1_past)
        theta2_cur = correcct_degree(float(parts[1]), theta2_past)
        plot(theta1_past, theta2_past, theta1_cur, theta2_cur)
        theta1_past, theta2_past = theta1_cur, theta2_cur
#-----------------------------------

if __name__ == '__main__':
    main()