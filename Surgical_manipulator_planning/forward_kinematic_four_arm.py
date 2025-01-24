import numpy as np 
import matplotlib.pyplot as plt

Ls = 23
Lg1 = 30
Lg2 = 40

def continumm_arm(Ls, theta):
    if theta == 0:
        return Ls * np.array([0, 0, 1])
    
    return (Ls / theta) * np.array([1.0 - np.cos(theta), 0, 
                                    np.sin(theta)])
    
def rigid_arm(Lg):
    return Lg * np.array([0, 0, 1])

def rotation_matrix_k(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rotation_matrix_j(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def forward_kinematic(q):
    P1 = rotation_matrix_k(q[1]) @ continumm_arm(Ls, q[0])
    P2 = P1 + rotation_matrix_k(q[1]) @ rotation_matrix_j(q[0]) @ rotation_matrix_k(-q[1]) @ rigid_arm(Lg1)
    P3 = P2 + rotation_matrix_k(q[1]) @ rotation_matrix_j(q[0]) @ rotation_matrix_k(-q[1]) @ rotation_matrix_k(q[3]) @ continumm_arm(Ls, q[2])
    P4 = P3 + rotation_matrix_k(q[1]) @ rotation_matrix_j(q[0]) @ rotation_matrix_k(-q[1]) @ rotation_matrix_k(q[3]) @ rotation_matrix_j(q[2]) @ rotation_matrix_k(-q[3]) @ rigid_arm(Lg2)
    
    return np.array([[0, 0, 0], P1, P2, P3, P4])
    

"""
q = np.array([np.pi / 3, np.pi , np.pi / 2.5, np.pi / 3])
locations = forward_kinematic(q) 
print(locations)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(locations[:,0], locations[:,1], locations[:,2], 'ro-')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
"""






