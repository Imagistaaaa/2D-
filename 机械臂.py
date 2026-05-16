import numpy as np
import matplotlib.pyplot as plt

theta1_past = theta2_past = 0.0
l1 = 1.0
l2 = 1.0
step = 50
origin = (0, 0)

def forward_kinematics(theta1, theta2):
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return origin, (x1, y1), (x2, y2)

def generate_trajectory(theta1_last, theta2_last, theta1_new, theta2_new):
    theta1s = np.linspace(theta1_last, theta1_new, num=step)
    theta2s = np.linspace(theta2_last, theta2_new, num=step)
    trail = []
    for t1, t2 in zip(theta1s, theta2s):
        trail.append(forward_kinematics(t1, t2)[2])
    return trail

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
        theta1_cur = np.radians(float(parts[0]))
        theta2_cur = np.radians(float(parts[1]))
        plot(theta1_past, theta2_past, theta1_cur, theta2_cur)
        theta1_past, theta2_past = theta1_cur, theta2_cur

if __name__ == '__main__':
    main()