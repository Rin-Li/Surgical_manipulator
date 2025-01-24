import numpy as np 

class collision_check:
    
    def __init__(self, Node_old, Node_new, centers, r):
        self.Node_old = Node_old
        self.centers = centers  # List of centers
        self.r = r
        self.Node_new = Node_new
    
    def collision(self):
        
        segment_1 = self.Node_new - self.Node_old
        
        
        for center, r in zip(self.centers, self.r):  # Iterate over each center

            point_start_vector = center - self.Node_old
            t1 = np.dot(point_start_vector, segment_1) / np.dot(segment_1, segment_1)
            
            if t1 < 0:
                closest_point_1 = self.Node_old
            elif t1 > 1:
                closest_point_1 = self.Node_new
            else:
                closest_point_1 = self.Node_old + t1 * segment_1
                
            min_distance_1 = np.linalg.norm(center - closest_point_1)
            
            if min_distance_1 < r + 3:
                return False  # Collision detected, return False
        
        return True  # No collision detected with any center


