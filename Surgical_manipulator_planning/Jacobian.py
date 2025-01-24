import numpy as np 
import sympy as sp
from DLS_inverse import DLS


Ls_val = 23
Lg1_val = 30
Lg2_val = 40


class weight_jacobian:
    def __init__(self, q, last_gradient, nearest_point_info, arm):
        self.q = q
        self.last_gradient = last_gradient
        self.nearest_point_info = nearest_point_info
        self.arm = arm
        self._last_gradient = self.last_gradient
    
    def Jacobian(self):
        theta1, delta1,theta2, delta2 = sp.symbols('theta1 delta1 theta2 delta2')
        Ls, Lg1, Lg2 = sp.symbols('Ls Lg1 Lg2')
        
        #旋转矩阵
        def rotation_matrix_k(delta):
            return sp.Matrix([
            [sp.cos(delta), -sp.sin(delta), 0],
            [sp.sin(delta), sp.cos(delta), 0],
            [0, 0, 1]
            ])
        #旋转矩阵
        def rotation_matrix_j(theta):
            return sp.Matrix([
            [sp.cos(theta), 0, sp.sin(theta)],
            [0, 1, 0],
            [-sp.sin(theta), 0, sp.cos(theta)]
            ])
        def continumm_arm(Ls, theta):
            return (Ls / theta) *sp.Matrix([1.0 - sp.cos(theta), 0, sp.sin(theta)])
        def rigid_arm(Lg):
            return sp.Matrix([0, 0, Lg])
        
        self.P1 = rotation_matrix_k(delta1) * continumm_arm(Ls, theta1)
        self.P2 = self.P1 + rotation_matrix_k(delta1) * rotation_matrix_j(theta1) * rotation_matrix_k(-delta1) * rigid_arm(Lg1)
        self.P3 = self.P2 + rotation_matrix_k(delta1) * rotation_matrix_j(theta1) * rotation_matrix_k(-delta1) * rotation_matrix_k(delta2) * continumm_arm(Ls, theta2)
        self.P4 = self.P3 + rotation_matrix_k(delta1) * rotation_matrix_j(theta1) * rotation_matrix_k(-delta1) * rotation_matrix_k(delta2) * rotation_matrix_j(theta2) * rotation_matrix_k(-delta2) * rigid_arm(Lg2)
        jacobian_matrix = sp.Matrix([
        [sp.diff(self.P4[i], var) for var in [theta1, delta1, theta2, delta2]]
        for i in range(3)
    ])
        jacobian_matrix_subs = jacobian_matrix.subs({Ls: Ls_val, 
                                                Lg1: Lg1_val, 
                                                Lg2: Lg2_val, 
                                                theta1: self.q[0], 
                                                delta1: self.q[1], 
                                                theta2: self.q[2], 
                                                delta2: self.q[3]})
        self.jacobian_matrix = np.array(jacobian_matrix_subs.evalf())
        
    def Jacobian_nearestpoint(self):
        theta1, delta1, theta2, delta2, nearest_theta = sp.symbols('theta1 delta1 theta2 delta2 nearest_theta')
        Ls, Lg1, Lg_nearest = sp.symbols('Ls Lg1 Lg_nearest')
        
        def rotation_matrix_k(delta):
            return sp.Matrix([
            [sp.cos(delta), -sp.sin(delta), 0],
            [sp.sin(delta), sp.cos(delta), 0],
            [0, 0, 1]
            ])
        
        def rotation_matrix_j(theta):
            return sp.Matrix([
            [sp.cos(theta), 0, sp.sin(theta)],
            [0, 1, 0],
            [-sp.sin(theta), 0, sp.cos(theta)]
            ])
        def continumm_arm(Ls, theta, nearest_theta):
            return (Ls / theta) *sp.Matrix([1.0 - sp.cos(nearest_theta), 0, sp.sin(nearest_theta)])
        
        def rigid_arm_nearest(Lg_nearest):
            return sp.Matrix([0, 0, Lg_nearest])
        

        
        if self.arm  == 1 and self.nearest_point_info != 0:
            p1 = rotation_matrix_k(delta1) * continumm_arm(Ls, theta1, nearest_theta)
            jacobian_nearest_matrix = sp.Matrix([
                [sp.diff(p1[0], theta1) + sp.diff(p1[0], nearest_theta), sp.diff(p1[0], delta1), 0, 0],
                [sp.diff(p1[1], theta1) + sp.diff(p1[1], nearest_theta), sp.diff(p1[1], delta1), 0, 0],
                [sp.diff(p1[2], theta1) + sp.diff(p1[2], nearest_theta), sp.diff(p1[2], delta1), 0, 0]
            ])
            jacobian_nearest_matrix_subs = jacobian_nearest_matrix.subs({Ls: Ls_val,
                                                                         nearest_theta: self.nearest_point_info,
                                                                         theta1: self.q[0],
                                                                         delta1: self.q[1]})
      
        elif self.arm == 2 and self.nearest_point_info == 0:
            jacobian_nearest_matrix = sp.Matrix([
                [sp.diff(self.P1[0], theta1), sp.diff(self.P1[0], delta1), 0, 0],
                [sp.diff(self.P1[1], theta1), sp.diff(self.P1[1], delta1), 0, 0],
                [sp.diff(self.P1[2], theta1), sp.diff(self.P1[2], delta1), 0, 0]
            ])
            jacobian_nearest_matrix_subs = jacobian_nearest_matrix.subs({Ls: Ls_val,
                                                                         theta1: self.q[0],
                                                                         delta1: self.q[1]})
        #这里是当最近点在第二个关节rigid关节上的情况，并且最近点不是端点，不是开始点
        elif self.arm == 2 and self.nearest_point_info != 0:
            p2 = self.P1 + rotation_matrix_k(delta1) * rotation_matrix_j(theta1) * rotation_matrix_k(-delta1) * rigid_arm_nearest(Lg_nearest)
            jacobian_nearest_matrix = sp.Matrix([
                [sp.diff(p2[0], theta1), sp.diff(p2[0], delta1), 0, 0],
                [sp.diff(p2[1], theta1), sp.diff(p2[1], delta1), 0, 0],
                [sp.diff(p2[2], theta1), sp.diff(p2[2], delta1), 0, 0]
            ])
            jacobian_nearest_matrix_subs = jacobian_nearest_matrix.subs({Ls: Ls_val,
                                                                            Lg_nearest: self.nearest_point_info,
                                                                            theta1: self.q[0],
                                                                            delta1: self.q[1]})
       
        elif self.arm == 3 and self.nearest_point_info == 0:
            jacobian_nearest_matrix = sp.Matrix([
                [sp.diff(self.P2[0], theta1), sp.diff(self.P2[0], delta1), 0, 0],
                [sp.diff(self.P2[1], theta1), sp.diff(self.P2[1], delta1), 0, 0],
                [sp.diff(self.P2[2], theta1), sp.diff(self.P2[2], delta1), 0, 0]
            ])
            jacobian_nearest_matrix_subs = jacobian_nearest_matrix.subs({Ls: Ls_val,
                                                                         Lg1: Lg1_val,
                                                                         theta1: self.q[0],
                                                                         delta1: self.q[1],
                                                                         })
       
        elif self.arm == 3 and self.nearest_point_info != 0:
            p3 = self.P2 + rotation_matrix_k(delta1) * rotation_matrix_j(theta1) * rotation_matrix_k(-delta1) * rotation_matrix_k(delta2) * continumm_arm(Ls, theta2, nearest_theta)
            jacobian_nearest_matrix = sp.Matrix([
                [sp.diff(p3[0], theta1), sp.diff(p3[0], delta1), sp.diff(p3[0], theta2) + sp.diff(p3[0], nearest_theta), sp.diff(p3[0], delta2)],
                [sp.diff(p3[1], theta1), sp.diff(p3[1], delta1), sp.diff(p3[1], theta2) + sp.diff(p3[1], nearest_theta), sp.diff(p3[1], delta2)],
                [sp.diff(p3[2], theta1), sp.diff(p3[2], delta1), sp.diff(p3[2], theta2) + sp.diff(p3[2], nearest_theta), sp.diff(p3[2], delta2)]
            ])
            jacobian_nearest_matrix_subs = jacobian_nearest_matrix.subs({Ls: Ls_val,
                                                                         Lg1: Lg1_val,
                                                                         theta1: self.q[0],
                                                                         delta1: self.q[1],
                                                                         theta2: self.q[2],
                                                                         delta2: self.q[3],
                                                                         nearest_theta: self.nearest_point_info
                                                                         })
      
        elif self.arm == 4 and self.nearest_point_info == 0:
            jacobian_nearest_matrix = sp.Matrix([
                [sp.diff(self.P3[0], theta1), sp.diff(self.P3[0], delta1), sp.diff(self.P3[0], theta2), sp.diff(self.P3[0], delta2)],
                [sp.diff(self.P3[1], theta1), sp.diff(self.P3[1], delta1), sp.diff(self.P3[1], theta2), sp.diff(self.P3[1], delta2)],
                [sp.diff(self.P3[2], theta1), sp.diff(self.P3[2], delta1), sp.diff(self.P3[2], theta2), sp.diff(self.P3[2], delta2)]
            ])
            jacobian_nearest_matrix_subs = jacobian_nearest_matrix.subs({Ls: Ls_val,
                                                                        Lg1: Lg1_val,
                                                                        theta1: self.q[0],
                                                                        delta1: self.q[1],
                                                                        theta2: self.q[2],
                                                                        delta2: self.q[3]
                                                                        })
        elif self.arm == 4 and self.nearest_point_info != 0:
            p4 = self.P3 + rotation_matrix_k(delta1) * rotation_matrix_j(theta1) * rotation_matrix_k(-delta1) * rotation_matrix_k(delta2) * rotation_matrix_j(theta2) * rotation_matrix_k(-delta2) * rigid_arm_nearest(Lg_nearest)
            jacobian_nearest_matrix = sp.Matrix([
                [sp.diff(p4[0], theta1), sp.diff(p4[0], delta1), sp.diff(p4[0], theta2), sp.diff(p4[0], delta2)],
                [sp.diff(p4[1], theta1), sp.diff(p4[1], delta1), sp.diff(p4[1], theta2), sp.diff(p4[1], delta2)],
                [sp.diff(p4[2], theta1), sp.diff(p4[2], delta1), sp.diff(p4[2], theta2), sp.diff(p4[2], delta2)]
            ])
            jacobian_nearest_matrix_subs = jacobian_nearest_matrix.subs({Ls: Ls_val,
                                                                        Lg1: Lg1_val,
                                                                        Lg_nearest: self.nearest_point_info,
                                                                        theta1: self.q[0],
                                                                        delta1: self.q[1],
                                                                        theta2: self.q[2],
                                                                        delta2: self.q[3]
                                                                        })
        self.jacobian_nearest_matrix = np.array(jacobian_nearest_matrix_subs.evalf())
    #Get the weight jacboian matrix 
    def Weight(self):
        #Symbol
        theta1, delta1, theta2, delta2 = sp.symbols('theta1 delta1 theta2 delta2')
        
        q_max = sp.Matrix([0.95 * np.pi / 2  , 0.95 * 2 * np.pi, 0.95 * np.pi / 2 , 0.95 * 2 * np.pi])
        q_min = sp.Matrix([0.95 * -np.pi / 2 , 0.0, 0.95 * 2-np.pi / 2, 0.0])
        Hq1 = (q_max[0] - q_min[0])**2 / ((q_max[0] - theta1) * (theta1 - q_min[0]))
        Hq2 = (q_max[1] - q_min[1])**2 / ((q_max[1] - delta1) * (delta1 - q_min[1]))
        Hq3 = (q_max[2] - q_min[2])**2 / ((q_max[2] - theta2) * (theta2 - q_min[2]))
        Hq4 = (q_max[3] - q_min[3])**2 / ((q_max[3] - delta2) * (delta2 - q_min[3]))
        Hq = (Hq1 + Hq2 + Hq3 + Hq4) / 4
        
        #Diff
        dHq_dtheta1 = sp.diff(Hq, theta1)
        dHq_dtheta2 = sp.diff(Hq, theta2)
        dHq_ddelta1 = sp.diff(Hq, delta1)
        dHq_ddelta2 = sp.diff(Hq, delta2)
        dHq_dd = sp.Matrix([dHq_dtheta1, dHq_ddelta1, dHq_dtheta2, dHq_ddelta2])
       
        #Replace
        
        dHq_dd = dHq_dd.subs({
            theta1: self.q[0],
            delta1: self.q[1],
            theta2: self.q[2],
            delta2: self.q[3]
        })
        

        
        W_diagonal_elements = np.zeros(4)
        dHq_dd = sp.Abs(dHq_dd)
        dHq_dd = np.array(dHq_dd.evalf())
        dHq_dd = dHq_dd.squeeze()

        if self._last_gradient is None:
            W_diagonal_elements = np.array([1 + dHq_dd[0], 1+ dHq_dd[1], 1 + dHq_dd[2], 1 + dHq_dd[3]])
            
        else:
            for i in range(4):
            
                if dHq_dd[i] - self._last_gradient[i] >= 0:
                    W_diagonal_elements[i] = 1 + dHq_dd[i]
                else:
                    W_diagonal_elements[i] = 1
                W_diagonal_elements = np.array(W_diagonal_elements)
            
        self._last_gradient = dHq_dd
        
        self.W_diagonal_elements = np.array(W_diagonal_elements, dtype=float)
        
        
        W_inver_diagonal = np.sqrt(1 / self.W_diagonal_elements)
        W_inver_diagonal = W_inver_diagonal.squeeze()
        W_inver = np.diag(W_inver_diagonal)
        self.W_inver = W_inver
    
    def combine_weight_jacobian_end(self):
 
        #Weighted Jacobian
        self.Jw = self.jacobian_matrix @ self.W_inver
        
    def psudo_inverse_end(self):
        self.Jw = np.array(self.Jw, dtype=np.float64)
        self.jacobian_matrix = np.array(self.jacobian_matrix, dtype = np.float64)
        self.psudo_inverse_jw = np.linalg.pinv(self.Jw) 
        self.psudo_inverse_j = np.linalg.pinv(self.jacobian_matrix)
    
    def combine_weight_jacobian_nearest(self):
        self.Jwo = self.jacobian_nearest_matrix @ self.W_inver
        self.Jwo = np.array(self.Jwo, dtype=np.float64)
        
    
    #Return the jacobian matrix and the weight matrix         
    def main_jacobian(self):
        self.Jacobian()  
        self.Weight()
        self.combine_weight_jacobian_end()
        self.psudo_inverse_end()

        
        
        return self.psudo_inverse_jw, self.psudo_inverse_j, self.W_inver, self.jacobian_matrix, self._last_gradient
    
    def main_jacobian_nearest(self):
        self.Jacobian()
        self.Jacobian_nearestpoint()
        self.Weight()
        self.combine_weight_jacobian_nearest()
        self.jacobian_nearest_matrix = np.array(self.jacobian_nearest_matrix, dtype=np.float64)

        
        return self.jacobian_nearest_matrix, self.Jwo
        

        
        
def main():
    theta_nearest_point = 0
    arm = 3
    q1 = np.array([ 0.03, 0.99 * 2 * np.pi , 0.04, 0.99 * 2 * np.pi])
    q2 = np.array([0.06, 0.98 * 2 * np.pi, 0.02, np.pi / 7])
    jacob = weight_jacobian(q1, None, theta_nearest_point, arm)
    a, b, c, d, e= jacob.main_jacobian()
    f, g = jacob.main_jacobian_nearest()
    print('W', c)

    jacob1 = weight_jacobian(q2, e, theta_nearest_point, arm)
    a,b,c,d,e = jacob1.main_jacobian()
    print('W', c)


    delta1 = 0.99 * 2 * np.pi


    a = 0.5/(delta1*(1 - 0.159154943091895*delta1)**2) - 19.7392088021787/(delta1**2*(6.28318530717959 - delta1))
    print("a:", a)

    b = -(1/2) * (((2 * np.pi - 2 * delta1) * (2 *np.pi)**2) / (((2 * np.pi - delta1)** 2)*(delta1**2)))
    print("b:", b)
    
if __name__ == '__main__':
    main()
