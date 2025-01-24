import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DCPQuery:
    def __init__(self):
        pass

    class Result:
        def __init__(self):
            self.distance = 0.0
            self.sqrDistance = 0.0
            self.closest = [np.zeros(3), np.zeros(3)]
            self.equidistant = False

    def __call__(self, point, circle):
        result = self.Result()

        PmC = point - circle['center']
        U, V, N = np.zeros(3), np.zeros(3), circle['normal']
        self.compute_orthogonal_basis(1, N, U, V)
        scaledQmC = (np.dot(N, N) * np.dot(U, PmC)) * U + np.dot(V, PmC) * V
        length_scaledQmC = np.linalg.norm(scaledQmC)
        
        if length_scaledQmC > 0:
            result.closest[0] = point
            result.closest[1] = circle['center'] + circle['radius'] * (scaledQmC / length_scaledQmC)
            height = np.dot(N, PmC)
            radial = np.linalg.norm(np.cross(N, PmC)) - circle['radius']
            result.sqrDistance = height * height + radial * radial
            result.distance = np.sqrt(result.sqrDistance)
            result.equidistant = False
        else:
            result.closest[0] = point
            result.closest[1] = circle['center'] + circle['radius'] * self.get_orthogonal(N)
            result.sqrDistance = np.dot(PmC, PmC) + circle['radius'] * circle['radius']
            result.distance = np.sqrt(result.sqrDistance)
            result.equidistant = True

        return result

    def compute_orthogonal_basis(self, num_inputs, v0, v1, v2):
        if num_inputs == 1:
            if abs(v0[0]) > abs(v0[1]):
                v1[:] = [-v0[2], 0, v0[0]]
            else:
                v1[:] = [0, v0[2], -v0[1]]
        else:
            v1[:] = np.dot(v0, v0) * v1 - np.dot(v1, v0) * v0

        if np.all(v1 == 0):
            v2[:] = np.zeros(3)
            return False

        v2[:] = np.cross(v0, v1)
        return not np.all(v2 == 0)

    def get_orthogonal(self, vector):
        if abs(vector[0]) > abs(vector[1]):
            return np.array([-vector[2], 0, vector[0]])
        else:
            return np.array([0, vector[2], -vector[1]])
def main():
    def plot_point_circle_result(point, circle, result):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the circle
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = circle['center'][0] + circle['radius'] * np.cos(theta)
        circle_y = circle['center'][1] + circle['radius'] * np.sin(theta)
        circle_z = np.full_like(circle_x, circle['center'][2])

        ax.plot(circle_x, circle_y, circle_z, 'b-', label='Circle')

        # Plot the original point
        ax.scatter(point[0], point[1], point[2], color='r', label='Point')

        # Plot the closest point on the circle
        ax.scatter(result.closest[1][0], result.closest[1][1], result.closest[1][2], color='g', label='Closest Point')

        # Draw a line between the point and the closest point on the circle
        ax.plot([point[0], result.closest[1][0]], 
                [point[1], result.closest[1][1]], 
                [point[2], result.closest[1][2]], 'k--', label='Distance')

        # Set labels
        
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        ax.legend()
        plt.show()

    # Example usage
    point = np.array([1.0, 2.0, 3.0])
    circle = {
        'center': np.array([0.0, 0.0, 0.0]),
        'normal': np.array([0.0, 0.0, 1.0]),
        'radius': 1.0
    }

    query = DCPQuery()
    result = query(point, circle)
    print(result.closest)

    plot_point_circle_result(point, circle, result)

if __name__ == '__main__':
    main()