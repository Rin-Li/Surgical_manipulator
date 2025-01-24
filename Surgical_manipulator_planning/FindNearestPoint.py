import numpy as np 
from shapely.geometry import LineString
from forward_kinematic import forward_kinematic, rotation_matrix_k, rotation_matrix_j
from NearestPointCircle import DCPQuery
Ls = 23
Lg1 = 30
Lg2 = 40

class NearestPoint:
    def __init__(self, q, obstacles_radius, obstacles_center):
        self.q = q
        self.obstacle_radius = obstacles_radius
        self.obstacle_center = obstacles_center
        self.arm_location = forward_kinematic(q)
        self.min_point_circle = []
        
        
    #Calcualte the center of two continumm arm and the radius
    def cicrle_point(self):
        self.center1 = rotation_matrix_k(self.q[1]) @ np.array([Ls / self.q[0], 0, 0]) #First
        self.arc1 = Ls / self.q[0]
        #Second
        self.center2 = self.arm_location[2] + rotation_matrix_k(self.q[1]) @ rotation_matrix_j(self.q[0]) @ rotation_matrix_k(-self.q[1]) @ rotation_matrix_k(self.q[3]) @ np.array([Ls / self.q[2], 0, 0])
        self.arc2 = Ls / self.q[2]
        
    
    #Calculate the middle point of two continumm arm
    def middle_point(self):
        self.middle_point_arm_1 = rotation_matrix_k(self.q[1]) @ ((Ls / self.q[0]) * np.array([1.0 - np.cos(self.q[0] / 2)
                                                                                               , 0, np.sin(self.q[0] / 2)]))
        self.middle_point_arm_2 = self.arm_location[2] + rotation_matrix_k(self.q[1]) @ rotation_matrix_j(self.q[0]) @ rotation_matrix_k(-self.q[1]) @ rotation_matrix_k(self.q[3]) @ ((Ls / self.q[2]) * np.array([1.0 - np.cos(self.q[2] / 2)
                                                                                                                                                                                                                    , 0, np.sin(self.q[2] / 2)]))
    
    def normal_vector(self):
        self.normal_arm_1 = rotation_matrix_k(self.q[1]) @ np.array([0.0, 1.0, 0.0])
        self.normal_arm_2 = rotation_matrix_k(self.q[1]) @ rotation_matrix_j(self.q[0]) @ rotation_matrix_k(-self.q[1]) @ rotation_matrix_k(self.q[3]) @ np.array([0.0, 1.0, 0.0])
    
    #Find the point is in the arc or not
    def where_is_point(self, arc_center, arc_start, arc_end, point, theta_all):
        #Find two line corrensponding to the start and end of the arc
        
        line_center_to_point = point - arc_center
        arc_start_to_center = arc_start - arc_center
        arc_end_to_center = arc_end - arc_center
        theta1 = np.arccos(np.dot(line_center_to_point, arc_start_to_center) / (np.linalg.norm(line_center_to_point) * np.linalg.norm(arc_start_to_center)))
        theta2 = np.arccos(np.dot(line_center_to_point, arc_end_to_center) / (np.linalg.norm(line_center_to_point) * np.linalg.norm(arc_end_to_center)))

        #If the sum of the two angle is equal to the theta_all, then the point is in the arc
        if theta1 + theta2 >= theta_all - 0.01 and theta1 + theta2 <= theta_all + 0.01:
            is_any_intersect = True
            theta_of_point = np.arccos(np.dot(line_center_to_point, arc_start_to_center) 
                                       / (np.linalg.norm(line_center_to_point) * np.linalg.norm(arc_start_to_center)))
        else:
            is_any_intersect = False
            theta_of_point = None
            
        return is_any_intersect, theta_of_point                                                                                                                                                                                  
    
    #First, calculate the minimum distance between the two rigid arms and obstacles
    
    def min_distance_rigid(self, rigid_start, rigid_end, Lg, obstacle_center):
       
        
        #Two vector one for rigid arm and one for obstacle
        vector_rigid = rigid_end - rigid_start
        vector_obstacle = obstacle_center - rigid_start
            #Calculatte the angle between two vector
        t1 = np.dot(vector_rigid, vector_obstacle) / np.dot(vector_rigid, vector_rigid)
        #If t1 is less than 0, then the nearest point is the start of the rigid arm
        if t1 < 0:
            nearest_point_rigid = rigid_start
        elif t1 > 1:
            nearest_point_rigid = rigid_end
        else:
            nearest_point_rigid = rigid_start + t1 * vector_rigid
            #Calculate the distance between the nearest point and the center of the obstacle
        #If the distance is less than the radius of the obstacle, then the nearest point is the point on the rigid arm
        length = (np.linalg.norm(nearest_point_rigid - rigid_start) / np.linalg.norm(vector_rigid)) * Lg
        vector_from_obstacle = obstacle_center - nearest_point_rigid
        min_distance = np.linalg.norm(nearest_point_rigid - obstacle_center) 
        return min_distance, length, vector_from_obstacle, nearest_point_rigid
    
    #Second, calculate the minimum distance between the two continumm arms and obstacles
    def min_distance_continumm(self, arc_center, arc_radius, arc_start, arc_end, obstacles_center, theta, normal):
        #Create a circle with the center, radius and normal vector
        circle = {'center': arc_center, 
                  'radius': arc_radius, 
                  'normal': normal}
        
        dcp = DCPQuery()
        result_arm = dcp(obstacles_center, circle)
    
        min_distance, min_point_circle = result_arm.distance, result_arm.closest[1]
        
        judge, theta_of_point = self.where_is_point(arc_center, arc_start, arc_end, min_point_circle, theta)
        self.min_point_circle.append(min_point_circle)
        
        
         #if where_is_point return True, then the point is in the arc

        if judge:
            vector_from_obstacle = obstacles_center - min_point_circle
            min_distance = np.linalg.norm(vector_from_obstacle)
            
            return min_distance, theta_of_point, vector_from_obstacle, min_point_circle
            
        else:
            vector1 = arc_start - obstacles_center
            vector2 = arc_end - obstacles_center
            a = np.linalg.norm(vector1)
            b = np.linalg.norm(vector2)
            min_distance, theta, vector_from_obstacle, min_point = min((a, 0, vector1, arc_start), (b, theta, vector2, arc_end))

            return min_distance, theta, vector_from_obstacle, min_point
        
        

    
    def find_nearest_point(self):
        self.cicrle_point()
        self.middle_point()
        self.normal_vector()
        
        mindistance_rigid_1 = None
        mindistance_rigid_2 = None
        mindistance_continumm_1 = None
        mindistance_continumm_2 = None
        
        for i in range(len(self.obstacle_center)):
            
            #Rigid arm
            distance_rigid_1, length_1, vector_from_obstacle_rigid_1, point_1 = self.min_distance_rigid(self.arm_location[1], self.arm_location[2], Lg1, self.obstacle_center[i])
            distance_rigid_2, length_2, vector_from_obstacle_rigid_2, point_2 = self.min_distance_rigid(self.arm_location[3], self.arm_location[4], Lg2, self.obstacle_center[i])
            #Continumm arm
            distance_continumm_1, theta_1, vector_from_obstacle_continumm_1, point_3 = self.min_distance_continumm(self.center1, self.arc1, self.arm_location[0], self.arm_location[1], 
                                                                                                self.obstacle_center[i], self.q[0], self.normal_arm_1)
            distance_continumm_2, theta_2, vector_from_obstacle_continumm_2, point_4 = self.min_distance_continumm(self.center2, self.arc2, self.arm_location[2], self.arm_location[3], 
                                                                                                self.obstacle_center[i], self.q[2], self.normal_arm_2)
            #Find the minimum distance
            if mindistance_rigid_1 is None or distance_rigid_1 < mindistance_rigid_1:
                mindistance_rigid_1 = distance_rigid_1
                min_length_1 = length_1
                min_vector_from_obstacle_rigid_1 = vector_from_obstacle_rigid_1
                min_point_1 = point_1
            if mindistance_rigid_2 is None or distance_rigid_2 < mindistance_rigid_2:
                mindistance_rigid_2 = distance_rigid_2
                min_length_2 = length_2
                min_vector_from_obstacle_rigid_2 = vector_from_obstacle_rigid_2
                min_point_2 = point_2
                
            if mindistance_continumm_1 is None or distance_continumm_1 < mindistance_continumm_1:
                mindistance_continumm_1 = distance_continumm_1
                min_theta_1 = theta_1
                min_vector_from_obstacle_continumm_1 = vector_from_obstacle_continumm_1
                min_point_3 = point_3
            if mindistance_continumm_2 is None or distance_continumm_2 < mindistance_continumm_2:
                mindistance_continumm_2 = distance_continumm_2
                min_theta_2 = theta_2
                min_vector_from_obstacle_continumm_2 = vector_from_obstacle_continumm_2
                min_point_4 = point_4
        
        return min((mindistance_rigid_1, min_length_1, min_vector_from_obstacle_rigid_1, 2, min_point_1), 
                   (mindistance_rigid_2, min_length_2, min_vector_from_obstacle_rigid_2, 4, min_point_2), 
                   (mindistance_continumm_1, min_theta_1, min_vector_from_obstacle_continumm_1, 1, min_point_3), 
                   (mindistance_continumm_2, min_theta_2, min_vector_from_obstacle_continumm_2, 3, min_point_4))
        





            
            
        
        
        
            
                                                                                                                                                                                                                     

    
    
    
            
            
            

    




        
        
        