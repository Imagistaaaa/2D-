import numpy as np

theta1  =  0.0
theta2  =  0.0
l1  =  1.0
l2  =  1.0

def forward_kinematics(theta1,theta2):
    '''
    Function
    ----------
    输入角度及臂长，返回两个关节的笛卡尔坐标

    Parameters
    ----------
    theta1 : float
        关节1角度
    theta2 : float
        关节2角度.
    l1 : float
        旋臂1长度
    l2 : float
        旋臂2长度.

    Returns
    -------
    float 2*(2)
        关节1和关节2的笛卡尔坐标

    '''
    x1  =  l1 * np.cos(theta1)
    y1  =  l1 * np.sin(theta1)
    x2  =  x1 + l2 * np.cos(theta1 + theta2)
    y2  =  y1 + l2 * np.sin(theta1 + theta2)
    return [(x1 , y1), (x2 , y2)]

def generate_trajectory(theta1_last, theta2_last, theta1_new, theta2_new, step):
    '''
    

    Parameters
    ----------
    theta1_last : TYPE
        DESCRIPTION.
    theta2_last : TYPE
        DESCRIPTION.
    theta1_new : TYPE
        DESCRIPTION.
    theta2_new : TYPE
        DESCRIPTION.
    step : TYPE
        DESCRIPTION.

    Returns
    -------
    trail : TYPE
        DESCRIPTION.

    '''
    theta1s  =  np.linspace(theta1_last, theta1_new, step = step)
    theta2s  =  np.linspace(theta2_last, theta2_new, step = step)
    trail=[]
    for t1,t2 in zip(theta1s,theta2s):
        trail.append(forward_kinematics(t1, t2)[1])
    return trail
