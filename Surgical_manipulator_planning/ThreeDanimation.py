import numpy as np 
import matplotlib.pyplot as plt
from forward_kinematic_four_arm import forward_kinematic, rotation_matrix_j, rotation_matrix_k
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import expm

Ls = 23
Lg1 = 30
Lg2 = 40

class RoboticArmAnimation:
    
    def __init__(self, q, path, obstacles_center, obstacles_radius):
        self.q = q
        self.path = path
        self.obstacles_center = obstacles_center
        self.obstacles_radius = obstacles_radius
        
  
    def circle_point(self, q, arm_location):
        center1 = rotation_matrix_k(q[1]) @ np.array([Ls / q[0], 0, 0]) # First
        # Second
        center2 = arm_location + rotation_matrix_k(q[1]) @ rotation_matrix_j(q[0]) @ rotation_matrix_k(-q[1]) @ rotation_matrix_k(q[3]) @ np.array([Ls / q[2], 0, 0])
        return center1, center2
    
    def plotAnimation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.path[:, 0], self.path[:, 1], self.path[:, 2], c='green', marker='x', markersize=8, label='Trajectory')
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)
        
        def plot_sphere(ax, center, radius, color='r'):
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            ax.plot_surface(x, y, z, color=color, alpha = 0.8) 
        
        for center, radius in zip(self.obstacles_center, self.obstacles_radius):
            plot_sphere(ax, center, radius, color='r')
        
        def get_sita(p):
            sita = np.arccos(np.dot(np.array((1, 0, 0)), p) / np.linalg.norm(p))
            if p[1] < 0:
                sita = 2 * np.pi - sita
            return sita
        
        def get_arc_points(center, p1, p2, step=0.01):
            center = np.array(center)
            p1 = np.array(p1)
            p2 = np.array(p2)
            R = np.sqrt(np.sum(np.power(p1 - center, 2)))
            cp = np.cross(center - p1, p2 - p1)
            a, b, c = cp
            d = np.dot(cp, center)
            cs = np.arccos(np.dot(p1 - center, p2 - center) / (np.linalg.norm(p1 - center) * np.linalg.norm(p2 - center)))
            roteAxis = np.cross(cp, [0, 0, 1])
            sita = np.arccos(np.dot(cp, [0, 0, 1]) / np.linalg.norm(cp))
            if get_sita(cp - np.array((0, 0, 1))) > 0:
                sita = -sita
            roteMatrix = expm(np.cross(np.eye(3), roteAxis / np.linalg.norm(roteAxis) * sita))
            roteBackMatrix = expm(np.cross(np.eye(3), roteAxis / np.linalg.norm(roteAxis) * (-sita)))
            P = np.vstack((center, p1, p2))
            RP = np.dot(P, roteMatrix)
            sp1 = get_sita(RP[1, :] - RP[0, :])
            sp2 = get_sita(RP[2, :] - RP[0, :])
            if np.abs(sp1 - sp2) > np.pi:
                st = np.hstack((np.arange(sp1, 2 * np.pi, step), np.arange(0, sp2, step))) if sp1 > sp2 else np.hstack(
                    (np.arange(sp2, 2 * np.pi, step), np.arange(0, sp1, step)))
            else:
                st = np.arange(sp1, sp2, step) if sp2 > sp1 else np.arange(sp2, sp1, step)
            arc = np.array((R * np.cos(st) + RP[0, 0], R * np.sin(st) + RP[0, 1], st * 0 + RP[0, 2]))
            for i in range(arc.shape[1]):
                arc[:, i] = np.dot(arc[:, i], roteBackMatrix)
            
            return arc
        
        def generate_cylinder_around_arc(arc_points, radius=1.0, num_segments = 100):
            surface_x = []
            surface_y = []
            surface_z = []

            # Create a circular cross-section
            theta = np.linspace(0, 2 * np.pi, num_segments)
            circle_x = radius * np.cos(theta)
            circle_y = radius * np.sin(theta)

            # Generate the surface along the arc
            for i in range(arc_points.shape[1]):
                point = arc_points[:, i]
                # Compute normal vectors perpendicular to the tangent
                if i < arc_points.shape[1] - 1:
                    tangent = arc_points[:, i + 1] - point
                else:
                    tangent = point - arc_points[:, i - 1]
                tangent = tangent / np.linalg.norm(tangent)
                
                # Choose a vector perpendicular to the tangent
                normal = np.array([0, 0, 1])
                if np.allclose(tangent, normal):
                    normal = np.array([1, 0, 0])
                
                binormal = np.cross(tangent, normal)
                binormal = binormal / np.linalg.norm(binormal)
                normal = np.cross(binormal, tangent)
                
                #  Add circular cross-section points to the surface
                circle_points = point.reshape(3, 1) + normal.reshape(3, 1) * circle_x + binormal.reshape(3, 1) * circle_y
                surface_x.append(circle_points[0, :])
                surface_y.append(circle_points[1, :])
                surface_z.append(circle_points[2, :])
            
            return np.array(surface_x), np.array(surface_y), np.array(surface_z)
        
        def plot_cylinder(ax, start_point, end_point, radius=2.0, num_segments=20):
            v = end_point - start_point
            mag = np.linalg.norm(v)
            v = v / mag
    
    # Create a vector not in the same direction as v
            not_v = np.array([1, 0, 0])
            if np.allclose(v, not_v):
                not_v = np.array([0, 1, 0])
    
    # Create two orthogonal vectors to v
            n1 = np.cross(v, not_v)
            n1 /= np.linalg.norm(n1)
            n2 = np.cross(v, n1)
    
    # Create the circular cross-section
            t = np.linspace(0, 2 * np.pi, num_segments)
            circle = (np.cos(t)[:, np.newaxis] * n1[np.newaxis, :] +
                      np.sin(t)[:, np.newaxis] * n2[np.newaxis, :]) * radius
    
    # Create the cylinder
            surface_x = []
            surface_y = []
            surface_z = []
            for i in np.linspace(0, mag, 2):  # two segments to cover start and end
                surface_x.append(start_point[0] + v[0] * i + circle[:, 0])
                surface_y.append(start_point[1] + v[1] * i + circle[:, 1])
                surface_z.append(start_point[2] + v[2] * i + circle[:, 2])
    
            ax.plot_surface(np.array(surface_x), np.array(surface_y), np.array(surface_z), color='limegreen', alpha=1.0)
            
        def plot_cylinder_with_hemisphere(ax, start_point, end_point, radius=2.0, num_segments=20):
    # Plot the cylinder
            v = end_point - start_point
            mag = np.linalg.norm(v)
            hemisphere_height = radius  
            cylinder_height = mag - hemisphere_height  

            new_end_point = start_point + v * (cylinder_height / mag)

            v_cylinder = new_end_point - start_point
            v_cylinder = v_cylinder / np.linalg.norm(v_cylinder)
    
    # Create a vector not in the same direction as v
            not_v = np.array([1, 0, 0])
            if np.allclose(v_cylinder, not_v):
                not_v = np.array([0, 1, 0])
    
    # Create two orthogonal vectors to v_cylinder
            n1 = np.cross(v_cylinder, not_v)
            n1 /= np.linalg.norm(n1)
            n2 = np.cross(v_cylinder, n1)
    
    # Ensure vectors are column vectors
            n1 = n1.reshape(-1, 1)
            n2 = n2.reshape(-1, 1)
            v_cylinder = v_cylinder.reshape(-1, 1)
    
    # Create the circular cross-section for the cylinder
            t = np.linspace(0, 2 * np.pi, num_segments)
            circle = (np.cos(t)[:, np.newaxis] * n1.T +
              np.sin(t)[:, np.newaxis] * n2.T) * radius
    
    # Create the cylinder
            surface_x = []
            surface_y = []
            surface_z = []
            for i in np.linspace(0, cylinder_height, 2):  # two segments to cover start and new_end
                surface_x.append(start_point[0] + v_cylinder[0] * i + circle[:, 0])
                surface_y.append(start_point[1] + v_cylinder[1] * i + circle[:, 1])
                surface_z.append(start_point[2] + v_cylinder[2] * i + circle[:, 2])
    
            ax.plot_surface(np.array(surface_x), np.array(surface_y), np.array(surface_z), color='limegreen', alpha = 1.0)
    
    # Plot the hemisphere at the new_end_point
            u = np.linspace(0, 2 * np.pi, num_segments)
            v_hemisphere = np.linspace(0, np.pi / 2, num_segments)  # Only half of the sphere
            hemisphere_x = radius * np.outer(np.cos(u), np.sin(v_hemisphere))
            hemisphere_y = radius * np.outer(np.sin(u), np.sin(v_hemisphere))
            hemisphere_z = radius * np.outer(np.ones(np.size(u)), np.cos(v_hemisphere))
    
    # Rotate hemisphere to align with cylinder axis
            hemisphere = np.array([hemisphere_x.ravel(), hemisphere_y.ravel(), hemisphere_z.ravel()])
            hemisphere_rotated = np.dot(hemisphere.T, np.hstack([n1, n2, v_cylinder])).T
    
    # Translate hemisphere to the new end of the cylinder
            hemisphere_rotated[0] += new_end_point[0]
            hemisphere_rotated[1] += new_end_point[1]
            hemisphere_rotated[2] += new_end_point[2]
    
            ax.plot_surface(hemisphere_rotated[0].reshape(num_segments, num_segments),
                    hemisphere_rotated[1].reshape(num_segments, num_segments),
                    hemisphere_rotated[2].reshape(num_segments, num_segments),
                    color='g', alpha = 1.0)

        
        def update(frame):
            ax.clear()
            ax.set_xlim([-100, 100])
            ax.set_ylim([-100, 100])
            ax.set_zlim([-100, 100])
            ax.set_xticks(np.arange(-100, 101, 50))
            ax.set_yticks(np.arange(-100, 101, 50))
            ax.set_zticks(np.arange(-100, 101, 50))
            ax.set_box_aspect([1, 1, 1])
            ax.plot(self.path[0][0], self.path[0][1], self.path[0][2], c='red', marker='x', markersize = 12)
            ax.plot(self.path[-1][0], self.path[-1][1], self.path[-1][2], c='red', marker='x', markersize = 12)
            ax.plot(self.path[1:-1, 0], self.path[1:-1, 1], self.path[1:-1, 2], c='peru', marker='x', markersize = 8)
            colors = ['tomato', 'cyan']
            for center, radius, color in zip(self.obstacles_center, self.obstacles_radius, colors):
                plot_sphere(ax, center, radius, color)
            q = self.q[frame]
            arm_location = forward_kinematic(q)
            center1, center2 = self.circle_point(q, arm_location[2])
            arc1 = get_arc_points(center1, arm_location[0], arm_location[1])
            arc2 = get_arc_points(center2, arm_location[2], arm_location[3])
            
            # Generate a cylinder with volume
            surface_x1, surface_y1, surface_z1 = generate_cylinder_around_arc(arc1, radius = 4.5)
            surface_x2, surface_y2, surface_z2 = generate_cylinder_around_arc(arc2, radius = 4.5)
            
            ax.plot_surface(surface_x1, surface_y1, surface_z1, color='gold', alpha = 1.0)
            ax.plot_surface(surface_x2, surface_y2, surface_z2, color='gold', alpha = 1.0)
            
            plot_cylinder(ax, arm_location[1], arm_location[2], radius=4.5)
            plot_cylinder_with_hemisphere(ax, arm_location[3], arm_location[4], radius = 4.5)
            ax.view_init(elev=21, azim=53)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            # plt.savefig("frame" + str(frame) + 'with' ".pdf", bbox_inches='tight')

        ani = FuncAnimation(fig, update, frames=len(self.q), repeat = True)
        plt.show()
        
        return ani
        

            
            
            
            
            
            

            
    
    
    
        
        
    