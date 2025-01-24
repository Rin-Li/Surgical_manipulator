import numpy as np 
from rrt_star import RRTStar 
from Jacobian import weight_jacobian 
from FindNearestPoint import NearestPoint
from inverse_kinematic import posture_without_restrict, posture_with_restrict
from direction import direction_dection

Ls = 23
Lg1 = 30
Lg2 = 40

def posture_without(q_init, x_start, x_goal, map_ranges, obstacles_center, obstacles_radius):
    #Get the path
    path = RRTStar(x_start, x_goal, map_ranges, obstacles_center, obstacles_radius).plan()
    print(path)
    
    
    #Ini gradient
    last_gradient = None
    #Ini q
    q = []
    q.append(q_init)
    for i in range(path.shape[0] - 1):
        print(i)
        #get min
        _, nearest_point_info, _, arm, _ = NearestPoint(q[i], obstacles_radius, obstacles_center).find_nearest_point()
        #Caclute the Jacobian
        Jwe_inv,_, W_inv, _, last_gradient = weight_jacobian(q[i], last_gradient, nearest_point_info, arm).main_jacobian()
        #Genertate the next posture
        q_tem = posture_without_restrict(path[i], path[i + 1], Jwe_inv, W_inv, q[i])
        
        q.append(q_tem)
    
    return q, path