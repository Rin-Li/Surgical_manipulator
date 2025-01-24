import numpy as np 
r = 30 + 18 + 4.5 
rmin = 30 + 12 + 4.5
rmax = 30 + 15 + 4.5 

def direction_dection(vector_from_obstacle, Jo, theta):
    
    x_vector = Jo @ theta
    vector_from_obstacle = np.array(vector_from_obstacle, dtype=np.float64)
    x_vector = np.array(x_vector, dtype=np.float64)
    angle = np.arccos(np.dot(vector_from_obstacle, x_vector) 
                      / (np.linalg.norm(vector_from_obstacle) * np.linalg.norm(x_vector)))
    
    distance = np.linalg.norm(vector_from_obstacle)
    unit_vector = vector_from_obstacle / distance
    
    r1 = distance - 4.5 - 30
    print('unit_vector',np.linalg.norm(unit_vector))
    d = -r1 * unit_vector
    print("disatnce,", distance)
    if distance < rmax and distance > rmin:
        av = ((distance - rmax) / (rmax - rmin)) ** 2
    elif distance < rmin:
        av = 1
    else:
        av = 0
    
    if distance < r and distance > rmax:
        ah = (1 / 2) * ( 1 - np.cos(np.pi * (distance - rmax) / (r - rmax)))
    elif distance < rmax:
        ah = 1
    else:
        ah = 0
    
    if angle < np.pi / 2 and distance < r :
        return d, av, ah
    else:
        return None, None, None
    
