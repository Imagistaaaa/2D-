import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
import heapq

#全局参数
theta1_past = theta2_past = 0.0
l1 = 1.5
l2 = 1.0
step = 50
origin = (0, 0)

#障碍物参数（圆心在可到达区域内，直径不超过内外半径之差 2.5-0.5=2.0）
OBSTACLE_CENTER = np.array([1.5, 1.0])
OBSTACLE_RADIUS = 0.6
#----------------------------------


#计算两个关节的坐标
def forward_kinematics(theta1, theta2):
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return origin, (x1, y1), (x2, y2)
#----------------------------------

#逆运动学部分（输入坐标以求解角度）
def backward_kinematics(x, y):

    L = np.sqrt(x**2 + y**2)

    cos_theta2 = (L**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)
    theta2_up = np.arccos(cos_theta2)
    theta2_down = -theta2_up

    for theta2 in [theta2_up, theta2_down]:
        k1 = l1 + l2 * np.cos(theta2)
        k2 = l2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        if theta2 == theta2_up:
            theta1_up = theta1
        else:
            theta1_down = theta1

    if abs(theta1_up-theta1_past)+abs(theta2_up-theta2_past) < abs(theta1_down-theta1_past)+abs(theta2_down-theta2_past):
        return theta1_up, theta2_up
    else:
        return theta1_down, theta2_down
#---------------------------------


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
def correct_degree(theta_cur,theta_past):
    if abs(theta_cur-theta_past)>np.pi:
        if theta_cur>theta_past:
            theta_cur = theta_cur - 2*np.pi
        else:
            theta_cur = theta_cur + 2*np.pi
    return theta_cur
#----------------------------------


#判断点是否在可到达区域内且不碰撞障碍物
def is_reachable(x, y):
    d_sq = x**2 + y**2
    if d_sq > (l1 + l2)**2 or d_sq < (l1 - l2)**2:
        return False
    obs_d_sq = (x - OBSTACLE_CENTER[0])**2 + (y - OBSTACLE_CENTER[1])**2
    if obs_d_sq < OBSTACLE_RADIUS**2:
        return False
    return True
#----------------------------------


#A* 网格寻路
def astar_path(start, goal, resolution=0.03):

    def world_to_grid(p):
        return (round(p[0] / resolution), round(p[1] / resolution))

    start_node = world_to_grid(start)
    goal_node = world_to_grid(goal)

    if start_node == goal_node:
        return [start, goal]

    #优先队列: (f, g, node)
    open_set = []
    heapq.heappush(open_set, (0.0, 0.0, start_node))

    came_from = {}
    g_score = {start_node: 0.0}

    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    max_iter = 50000
    iters = 0

    while open_set and iters < max_iter:
        iters += 1
        f, g, current = heapq.heappop(open_set)

        if g > g_score.get(current, float('inf')):
            continue

        if current == goal_node:
            #回溯路径
            path = [np.array(goal)]
            node = current
            while node in came_from:
                node = came_from[node]
                path.append(np.array([node[0] * resolution,
                                      node[1] * resolution]))
            path.reverse()
            return path

        cx, cy = current
        for dx, dy in neighbours:
            nx, ny = cx + dx, cy + dy
            neighbour = (nx, ny)
            wx, wy = nx * resolution, ny * resolution

            if not is_reachable(wx, wy):
                continue

            move_cost = np.sqrt(dx**2 + dy**2) * resolution
            tentative_g = g + move_cost

            if tentative_g < g_score.get(neighbour, float('inf')):
                g_score[neighbour] = tentative_g
                h = np.sqrt((nx - goal_node[0])**2 + (ny - goal_node[1])**2) * resolution
                f_new = tentative_g + h
                heapq.heappush(open_set, (f_new, tentative_g, neighbour))
                came_from[neighbour] = current

    return None  #无可行路径
#----------------------------------


#路径平滑（迭代拉直，保持避障）
def smooth_path(path, iterations=30):
    if len(path) <= 2:
        return path[:]
    smoothed = [p.copy() for p in path]
    for _ in range(iterations):
        for i in range(1, len(smoothed) - 1):
            mid = (smoothed[i - 1] + smoothed[i + 1]) / 2.0
            if is_reachable(mid[0], mid[1]):
                smoothed[i] = mid
    return smoothed
#----------------------------------


#综合绘图：障碍物、可到达区域边界、机械臂始末状态、路径
def plot_result(start_xy, goal_xy, path):
    global theta1_past, theta2_past

    #求起始点关节角
    theta1_past, theta2_past = 0.0, 0.0
    theta1_start, theta2_start = backward_kinematics(start_xy[0], start_xy[1])

    #求目标点关节角
    theta1_past, theta2_past = theta1_start, theta2_start
    theta1_goal, theta2_goal = backward_kinematics(goal_xy[0], goal_xy[1])

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    #可到达区域边界
    inner = MplCircle((0, 0), abs(l1 - l2), fill=False, linestyle='--',
                      edgecolor='gray', alpha=0.4)
    outer = MplCircle((0, 0), l1 + l2, fill=False, linestyle='--',
                      edgecolor='gray', alpha=0.4, label='reachable boundary')
    ax.add_patch(inner)
    ax.add_patch(outer)

    #障碍物
    obs = MplCircle(OBSTACLE_CENTER, OBSTACLE_RADIUS, fill=True,
                    color='red', alpha=0.35, label='obstacle')
    ax.add_patch(obs)

    #机械臂起始状态
    start_arm = np.array(forward_kinematics(theta1_start, theta2_start))
    ax.plot(start_arm[:, 0], start_arm[:, 1], color='b', linestyle='-.',
            linewidth=2, marker='o', markersize=4, markerfacecolor='w',
            markeredgecolor='b', label='arm start')

    #机械臂目标状态
    goal_arm = np.array(forward_kinematics(theta1_goal, theta2_goal))
    ax.plot(goal_arm[:, 0], goal_arm[:, 1], color='g', linestyle='-',
            linewidth=4, marker='o', markersize=6, markerfacecolor='w',
            markeredgecolor='g', label='arm goal')

    #路径
    if path is not None:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 0], path_arr[:, 1], color='orange', linestyle='-',
                linewidth=2, label='path')
        ax.scatter(*start_xy, color='blue', s=60, zorder=5, label='start point')
        ax.scatter(*goal_xy, color='green', s=60, zorder=5, label='goal point')

    ax.set_xlim(-1.25 * (l1 + l2), 1.25 * (l1 + l2))
    ax.set_ylim(-1.25 * (l1 + l2), 1.25 * (l1 + l2))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('2-DOF Arm Path Planning (A*)')

    plt.show()
#-----------------------------------


#主程序
def main():
    outer_r = l1 + l2
    inner_r = abs(l1 - l2)
    print(f'Reachable area: annulus  r_in={inner_r:.1f},  r_out={outer_r:.1f}')
    print(f'Obstacle:   center={tuple(OBSTACLE_CENTER)},  radius={OBSTACLE_RADIUS}')
    print()

    while True:
        inp = input('start_x start_y goal_x goal_y (q for quit): ').strip()
        if inp.lower() == 'q':
            break
        parts = inp.split()
        if len(parts) != 4:
            print('Expect 4 numbers: start_x start_y goal_x goal_y')
            continue

        sx, sy, gx, gy = map(float, parts)
        start = np.array([sx, sy])
        goal = np.array([gx, gy])

        if not is_reachable(sx, sy):
            print('Start point is unreachable (outside workspace or inside obstacle)')
            continue
        if not is_reachable(gx, gy):
            print('Goal point is unreachable (outside workspace or inside obstacle)')
            continue

        print('Searching path via A* ...')
        path = astar_path(start, goal)
        if path is None:
            print('No feasible path found.')
            continue

        path = smooth_path(path)
        print(f'Path found: {len(path)} waypoints')
        plot_result(start, goal, path)
#-----------------------------------

if __name__ == '__main__':
    main()
