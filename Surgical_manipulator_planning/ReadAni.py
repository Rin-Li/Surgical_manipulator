from ThreeDanimation import RoboticArmAnimation
import numpy as np 

#Read q, path
obstacles_center = [np.array([10, 40, 30]), np.array([40, -40, 50])]

    
obstacles_radius = [30, 30]
q_edit = np.load("q_edit.npy")
print(q_edit.shape)
min_point_edit = np.load("min_point_edit.npy")
for i in range(q_edit.shape[0]):
    print(i)
    print(q_edit[i])
    

path = np.load("path.npy")
RoboticArmAnimation(q_edit, path, obstacles_center, obstacles_radius).plotAnimation()
