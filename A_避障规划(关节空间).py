# -*- coding: utf-8 -*-
"""
2-DOF 机械臂避障路径规划（C-space / 关节空间 A* 版）

与早期笛卡尔空间末端寻路版本不同，本程序在关节空间 (theta1, theta2)
网格上做 A* 搜索。每个候选构型都会对大臂、小臂两条连杆分别做
"线段-圆"碰撞检测，因此规划出的整条运动过程中连杆不会穿透障碍物。
目标点的肘上/肘下两组逆解都会被尝试，自动选择能绕开障碍物、
且总路径更短的连杆构型方向。
"""

import numpy as np
import math
import matplotlib
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

#连杆安全间隙：连杆（视为线段）与障碍物边界需保持的最小距离
LINK_CLEARANCE = 0.03
#规划用膨胀间隙：搜索与平滑阶段额外多留出的余量，
#用于抵消"两构型之间直线插值"与实际扫掠轨迹的微小偏差
PLAN_CLEARANCE = LINK_CLEARANCE + 0.01

#插值碰撞检测的角度采样间距（弧度）
CHECK_SPACING = 0.01

#C空间网格：把 [-pi, pi) 均分为 N 份
CSPACE_GRID_N = 126
CSPACE_RES = 2.0 * np.pi / CSPACE_GRID_N
#----------------------------------


#计算两个关节的坐标
def forward_kinematics(theta1, theta2):
    x1 = l1 * math.cos(theta1)
    y1 = l1 * math.sin(theta1)
    x2 = x1 + l2 * math.cos(theta1 + theta2)
    y2 = y1 + l2 * math.sin(theta1 + theta2)
    return (0.0, 0.0), (x1, y1), (x2, y2)
#----------------------------------


#逆运动学：返回肘上/肘下两组解 [(t1_up, t2_up), (t1_down, t2_down)]
def ik_solutions(x, y):
    L = np.sqrt(x**2 + y**2)

    cos_theta2 = (L**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)
    theta2_up = np.arccos(cos_theta2)
    theta2_down = -theta2_up

    solutions = []
    for theta2 in [theta2_up, theta2_down]:
        k1 = l1 + l2 * np.cos(theta2)
        k2 = l2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        solutions.append((theta1, theta2))
    return solutions
#----------------------------------


#逆运动学（保持原接口）：在两组解中选相对上一次构型转动量更小的一组
def backward_kinematics(x, y):
    (theta1_up, theta2_up), (theta1_down, theta2_down) = ik_solutions(x, y)
    if abs(theta1_up - theta1_past) + abs(theta2_up - theta2_past) < \
       abs(theta1_down - theta1_past) + abs(theta2_down - theta2_past):
        return theta1_up, theta2_up
    else:
        return theta1_down, theta2_down
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
def correct_degree(theta_cur, theta_past):
    if abs(theta_cur - theta_past) > np.pi:
        if theta_cur > theta_past:
            theta_cur = theta_cur - 2 * np.pi
        else:
            theta_cur = theta_cur + 2 * np.pi
    return theta_cur
#----------------------------------


#判断末端点是否在可到达区域内且不碰撞障碍物
def is_reachable(x, y):
    d_sq = x**2 + y**2
    if d_sq > (l1 + l2)**2 or d_sq < (l1 - l2)**2:
        return False
    obs_d_sq = (x - OBSTACLE_CENTER[0])**2 + (y - OBSTACLE_CENTER[1])**2
    if obs_d_sq < OBSTACLE_RADIUS**2:
        return False
    return True
#----------------------------------


#线段 p1-p2 到点 c 的距离平方（纯标量运算，供高频调用）
def seg_point_dist_sq(p1, p2, c):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    fx = p1[0] - c[0]
    fy = p1[1] - c[1]
    a = dx * dx + dy * dy
    if a < 1e-12:
        return fx * fx + fy * fy
    t = -(fx * dx + fy * dy) / a
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    ex = fx + t * dx
    ey = fy + t * dy
    return ex * ex + ey * ey
#----------------------------------


#连杆碰撞检测：某构型下大臂、小臂两条线段是否侵入障碍物（含安全间隙）
def check_collision(theta1, theta2, clearance=LINK_CLEARANCE):
    _, elbow, end = forward_kinematics(theta1, theta2)
    r_eff_sq = (OBSTACLE_RADIUS + clearance) ** 2
    c = (OBSTACLE_CENTER[0], OBSTACLE_CENTER[1])
    #大臂：基座(0,0)-肘部
    if seg_point_dist_sq((0.0, 0.0), elbow, c) < r_eff_sq:
        return True
    #小臂：肘部-末端
    if seg_point_dist_sq(elbow, end, c) < r_eff_sq:
        return True
    return False
#----------------------------------


#C空间索引 <-> 角度
def theta_to_index(theta):
    return int(round((theta + np.pi) / CSPACE_RES)) % CSPACE_GRID_N

def index_to_theta(i):
    return -np.pi + i * CSPACE_RES
#----------------------------------


#构建C空间碰撞图：True 表示该构型碰撞
def build_cspace_grid():
    grid = np.zeros((CSPACE_GRID_N, CSPACE_GRID_N), dtype=bool)
    for i in range(CSPACE_GRID_N):
        t1 = index_to_theta(i)
        for j in range(CSPACE_GRID_N):
            grid[i, j] = check_collision(t1, index_to_theta(j))
    return grid
#----------------------------------


#A* 在关节空间网格上寻路（八邻域，角度维度环形回绕）
def astar_cspace(start_cfg, goal_cfg, cspace_grid=None, max_iter=200000):
    if cspace_grid is None:
        cspace_grid = build_cspace_grid()

    start_node = (theta_to_index(start_cfg[0]), theta_to_index(start_cfg[1]))
    goal_node = (theta_to_index(goal_cfg[0]), theta_to_index(goal_cfg[1]))

    if cspace_grid[start_node]:
        return None
    if cspace_grid[goal_node]:
        return None

    if start_node == goal_node:
        return [np.array(start_cfg), np.array(goal_cfg)]

    N = CSPACE_GRID_N

    def wrap_delta(a, b):
        d = abs(a - b)
        return min(d, N - d)

    def heuristic(node):
        return CSPACE_RES * math.hypot(wrap_delta(node[0], goal_node[0]),
                                       wrap_delta(node[1], goal_node[1]))

    open_set = []
    heapq.heappush(open_set, (heuristic(start_node), 0.0, start_node))

    came_from = {}
    g_score = {start_node: 0.0}

    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    iters = 0
    while open_set and iters < max_iter:
        iters += 1
        f, g, current = heapq.heappop(open_set)

        if g > g_score.get(current, float('inf')):
            continue

        if current == goal_node:
            #回溯路径
            path = [np.array(goal_cfg)]
            node = current
            while node in came_from:
                node = came_from[node]
                path.append(np.array([index_to_theta(node[0]),
                                      index_to_theta(node[1])]))
            path.reverse()
            return path

        ci, cj = current
        cur_cfg = (index_to_theta(ci), index_to_theta(cj))
        for di, dj in neighbours:
            ni, nj = (ci + di) % N, (cj + dj) % N
            neighbour = (ni, nj)

            if cspace_grid[neighbour]:
                continue

            #边级验证：当前构型与邻居构型之间的插值段也必须无碰
            nb_cfg = (index_to_theta(ni), index_to_theta(nj))
            if not segment_collision_free(cur_cfg, nb_cfg):
                continue

            move_cost = math.hypot(di, dj) * CSPACE_RES
            tentative_g = g + move_cost

            if tentative_g < g_score.get(neighbour, float('inf')):
                g_score[neighbour] = tentative_g
                f_new = tentative_g + heuristic(neighbour)
                heapq.heappush(open_set, (f_new, tentative_g, neighbour))
                came_from[neighbour] = current

    return None  #无可行路径
#----------------------------------


#两构型之间按角度线性插值的整段轨迹是否全程无碰撞
#（A* 网格节点本身无碰，不代表节点之间的插值段无碰，
#  尤其在障碍物边界切线附近，C空间碰撞区可能呈薄片状，必须逐段验证）
def segment_collision_free(c1, c2, clearance=PLAN_CLEARANCE,
                           spacing=CHECK_SPACING, max_samples=256):
    c1 = (float(c1[0]), float(c1[1]))
    c2 = (float(c2[0]), float(c2[1]))
    #角度解回绕：C空间两个维度都是环形的，
    #若两构型在某维度相差超过 pi，说明跨越了 ±pi 接缝，
    #插值必须沿短弧方向进行（碰撞检测对 2*pi 周期，平移不改变结果）
    for k in range(2):
        d = c2[k] - c1[k]
        if d > np.pi:
            c2 = (c2[0] - 2 * np.pi, c2[1]) if k == 0 else (c2[0], c2[1] - 2 * np.pi)
        elif d < -np.pi:
            c2 = (c2[0] + 2 * np.pi, c2[1]) if k == 0 else (c2[0], c2[1] + 2 * np.pi)
    d = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
    n = int(math.ceil(d / spacing)) + 1
    if n > max_samples:
        n = max_samples
    for k in range(n + 1):
        t = k / n
        cfg = (c1[0] + (c2[0] - c1[0]) * t, c1[1] + (c2[1] - c1[1]) * t)
        if check_collision(cfg[0], cfg[1], clearance):
            return False
    return True
#----------------------------------


#路径平滑（关节空间迭代拉直，平滑前后均保持连杆无碰撞）
def smooth_configs(path, iterations=30):
    if len(path) <= 2:
        return [p.copy() for p in path]
    smoothed = [p.copy() for p in path]
    for _ in range(iterations):
        for i in range(1, len(smoothed) - 1):
            mid = (smoothed[i - 1] + smoothed[i + 1]) / 2.0
            #中点及中点两侧的过渡段都必须无碰撞
            if check_collision(mid[0], mid[1]) \
               or not segment_collision_free(smoothed[i - 1], mid) \
               or not segment_collision_free(mid, smoothed[i + 1]):
                continue
            smoothed[i] = mid
    return smoothed
#----------------------------------


#路径角度解缠：消除相邻构型间 ±2pi 的跳变，使平滑和行程计算沿真实短弧进行
def unwrap_configs(path):
    out = [np.array(path[0], dtype=float)]
    for cfg in path[1:]:
        prev = out[-1]
        c = np.array(cfg, dtype=float)
        for k in range(2):
            while c[k] - prev[k] > np.pi:
                c[k] -= 2 * np.pi
            while c[k] - prev[k] < -np.pi:
                c[k] += 2 * np.pi
        out.append(c)
    return out
#----------------------------------


#综合规划入口：给定起止末端坐标，返回 (构型路径, 起止构型, 目标构型方向名)
def plan_path(start_xy, goal_xy, cspace_grid=None, verbose=True):
    global theta1_past, theta2_past

    sx, sy = start_xy
    gx, gy = goal_xy
    if not is_reachable(sx, sy):
        raise ValueError('Start point is unreachable (outside workspace or inside obstacle)')
    if not is_reachable(gx, gy):
        raise ValueError('Goal point is unreachable (outside workspace or inside obstacle)')

    #起始构型：优先沿用上次构型转动量最小的分支；若该分支连杆碰撞则自动换另一支
    start_solutions = ik_solutions(sx, sy)
    start_solutions.sort(key=lambda c: abs(c[0] - theta1_past) + abs(c[1] - theta2_past))
    start_cfg = None
    for cand in start_solutions:
        if not check_collision(*cand):
            start_cfg = np.array(cand)
            break
    if start_cfg is None:
        raise ValueError('Both IK branches of the start point collide with the obstacle via links')

    #目标构型：肘上/肘下两组解都参与搜索，自动选出可绕障且路径更短的连杆方向
    goal_solutions = ik_solutions(gx, gy)
    best = None  # (path_len, path, goal_cfg, branch_name)
    for cand, name in zip(goal_solutions, ['elbow-up', 'elbow-down']):
        if check_collision(*cand):
            if verbose:
                print(f'Goal branch {name}: blocked by link collision, skipped')
            continue
        raw = astar_cspace(start_cfg, np.array(cand), cspace_grid)
        if raw is None:
            if verbose:
                print(f'Goal branch {name}: no collision-free path in C-space')
            continue
        path = unwrap_configs(raw)
        #总关节行程作为路径长度度量
        arr = np.array(path)
        seg = np.linalg.norm(np.diff(arr, axis=0), axis=1).sum()
        if best is None or seg < best[0]:
            best = (seg, path, np.array(cand), name)

    if best is None:
        return None, (start_cfg, None, None)

    seg_len, path, goal_cfg, branch_name = best

    #最终安全校验：按真实安全间隙、更密的采样间距逐段验证，
    #不通过则退回未经平滑的原始 A* 路径（该路径的每条边已逐段验证过）
    def full_verify(p):
        for a, b in zip(p[:-1], p[1:]):
            if not segment_collision_free(a, b, clearance=LINK_CLEARANCE,
                                          spacing=0.002):
                return False
        return True

    smoothed = smooth_configs(path)
    if full_verify(smoothed):
        path = smoothed
    elif full_verify(path):
        path = path
    else:
        return None, (start_cfg, goal_cfg, branch_name)

    if verbose:
        print(f'Goal branch selected: {branch_name} (shorter joint-space path)')
    return path, (start_cfg, goal_cfg, branch_name)
#----------------------------------


#综合绘图：左图为工作空间（含连杆无碰验证的路径），右图为C空间碰撞图与搜索路径
def plot_result(start_xy, goal_xy, config_path, configs_info):
    start_cfg, goal_cfg, branch_name = configs_info

    fig, (ax, cax) = plt.subplots(1, 2, figsize=(14, 7), dpi=130)

    #---------- 左图：工作空间 ----------
    inner = MplCircle((0, 0), abs(l1 - l2), fill=False, linestyle='--',
                      edgecolor='gray', alpha=0.4)
    outer = MplCircle((0, 0), l1 + l2, fill=False, linestyle='--',
                      edgecolor='gray', alpha=0.4, label='reachable boundary')
    ax.add_patch(inner)
    ax.add_patch(outer)

    obs = MplCircle(OBSTACLE_CENTER, OBSTACLE_RADIUS, fill=True,
                    color='red', alpha=0.35, label='obstacle')
    ax.add_patch(obs)

    #机械臂起始状态
    start_arm = np.array(forward_kinematics(*start_cfg))
    ax.plot(start_arm[:, 0], start_arm[:, 1], color='b', linestyle='-.',
            linewidth=2, marker='o', markersize=4, markerfacecolor='w',
            markeredgecolor='b', label='arm start')

    #机械臂目标状态
    goal_arm = np.array(forward_kinematics(*goal_cfg))
    ax.plot(goal_arm[:, 0], goal_arm[:, 1], color='g', linestyle='-',
            linewidth=4, marker='o', markersize=6, markerfacecolor='w',
            markeredgecolor='g', label='arm goal')

    if config_path is not None:
        #路径中的中间构型：淡灰色细线，展示机械臂整体如何绕障
        n = len(config_path)
        show_step = max(1, n // 15)
        for k in range(show_step, n - show_step, show_step):
            arm = np.array(forward_kinematics(*config_path[k]))
            ax.plot(arm[:, 0], arm[:, 1], color='gray',
                    linewidth=0.8, alpha=0.35)
        ax.plot([], [], color='gray', linewidth=0.8, alpha=0.5,
                label='intermediate arm')

        #末端路径
        end_pts = np.array([forward_kinematics(*cfg)[2] for cfg in config_path])
        ax.plot(end_pts[:, 0], end_pts[:, 1], color='orange', linestyle='-',
                linewidth=2, label='end-effector path')

    ax.scatter(*start_xy, color='blue', s=60, zorder=5, label='start point')
    ax.scatter(*goal_xy, color='green', s=60, zorder=5, label='goal point')

    ax.set_xlim(-1.25 * (l1 + l2), 1.25 * (l1 + l2))
    ax.set_ylim(-1.25 * (l1 + l2), 1.25 * (l1 + l2))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Workspace (link collision checked)')

    #---------- 右图：C空间 ----------
    if config_path is not None:
        grid = build_cspace_grid()
        cax.imshow(grid.T.astype(int), origin='lower',
                   extent=[-np.pi, np.pi, -np.pi, np.pi],
                   cmap='Reds', alpha=0.6, aspect='auto')
        path_arr = np.array(config_path)
        cax.plot(path_arr[:, 0], path_arr[:, 1], color='deepskyblue',
                 linewidth=1.5, label='C-space path')
        cax.scatter(*start_cfg, color='blue', s=50, zorder=5, label='start cfg')
        cax.scatter(*goal_cfg, color='green', s=50, zorder=5, label='goal cfg')
        cax.legend(loc='upper right', fontsize=8)

    cax.set_xlim(-np.pi, np.pi)
    cax.set_ylim(-np.pi, np.pi)
    cax.set_xlabel('theta1 (rad)')
    cax.set_ylabel('theta2 (rad)')
    cax.set_title('C-space (red = link collision)')
    cax.grid(True, alpha=0.3)

    if branch_name:
        fig.suptitle(f'2-DOF Arm Path Planning (C-space A*)  —  goal branch: {branch_name}')
    else:
        fig.suptitle('2-DOF Arm Path Planning (C-space A*)  —  no feasible path')

    plt.tight_layout()
    plt.show()
#-----------------------------------


#主程序
def main():
    outer_r = l1 + l2
    inner_r = abs(l1 - l2)
    print(f'Reachable area: annulus  r_in={inner_r:.1f},  r_out={outer_r:.1f}')
    print(f'Obstacle:   center={tuple(OBSTACLE_CENTER)},  radius={OBSTACLE_RADIUS}')
    print('Planning in C-space (joint space) with full-link collision checking.')
    print()

    cspace_grid = build_cspace_grid()

    while True:
        inp = input('start_x start_y goal_x goal_y (q for quit): ').strip()
        if inp.lower() == 'q':
            break
        parts = inp.split()
        if len(parts) != 4:
            print('Expect 4 numbers: start_x start_y goal_x goal_y')
            continue

        try:
            sx, sy, gx, gy = map(float, parts)
            start = np.array([sx, sy])
            goal = np.array([gx, gy])

            print('Searching path via C-space A* ...')
            path, configs_info = plan_path(start, goal, cspace_grid)
        except ValueError as e:
            print(e)
            continue

        if path is None:
            print('No feasible path found (both elbow branches are blocked).')
            continue

        print(f'Path found: {len(path)} configurations')
        plot_result(start, goal, path, configs_info)

        #更新全局"上一次构型"，供下次规划选择起始分支
        global theta1_past, theta2_past
        theta1_past, theta2_past = path[-1]
#-----------------------------------

if __name__ == '__main__':
    main()
