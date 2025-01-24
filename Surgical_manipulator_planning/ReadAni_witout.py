from ThreeDanimation import RoboticArmAnimation
import numpy as np 

#Read q, path
obstacles_center = [np.array([10, 40, 30]), np.array([40, -40, 50])]

obstacles_radius = [30, 30]
q = np.load("q.npy")
print(q.shape[0])
for i in range(q.shape[0]):
    print(i)
    print(q[i])
min_point_no = np.load("min_point_no.npy")

path = np.load("path.npy")
RoboticArmAnimation(q, path, obstacles_center, obstacles_radius).plotAnimation()
