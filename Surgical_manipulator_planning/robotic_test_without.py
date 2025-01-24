import numpy as np 
from robotic_posture_without import posture_without
from ThreeDanimation import RoboticArmAnimation

q_init = [np.pi / 3, np.pi, np.pi / 2.5, np.pi / 3]
x_start = [ -50.06973358, 43.89813244,  70.85376038]
x_goal = [-58.326769,   -52.94270309,  14.11190423]
map_ranges = [[-90, 90], [-90, 90], [-90, 90]]
obstacles_center = [[0, -40, 10]]
obstacles_radius = [20]
q, path = posture_without(q_init, x_start, x_goal, map_ranges, obstacles_center, obstacles_radius)

RoboticArmAnimation(q, path, obstacles_center, obstacles_radius).plotAnimation()
