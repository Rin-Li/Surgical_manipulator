import numpy as np
import matplotlib.pyplot as plt
from collision_det import collision_check
from scipy.interpolate import splprep, splev
from mpl_toolkits.mplot3d import Axes3D

class Node:
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates)
        self.parent = None
        self.cost = 0.0
        


class RRTStar:
    def __init__(self, start, goal, map_ranges, center, radius, step_size = 5, max_iter=10000, goal_bias=0.2, search_radius = 1):
        self.start = Node(start)
        self.goal = Node(goal)
        self.map_ranges = map_ranges
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree = [self.start]
        self.goal_bias = goal_bias
        self.search_radius = search_radius
        self.center = center
        self.radius = radius

    def get_random_node(self):
        if np.random.rand() < self.goal_bias:
            return Node(self.goal.coordinates)  
        else:
            coordinates = [np.random.uniform(r[0], r[1]) for r in self.map_ranges]
            return Node(coordinates)

    def get_nearest_node(self, random_node):
        nearest_node = self.tree[0]
        min_dist = np.linalg.norm(nearest_node.coordinates - random_node.coordinates)
        for node in self.tree:
            dist = np.linalg.norm(node.coordinates - random_node.coordinates)
            if dist < min_dist:
                nearest_node = node
                min_dist = dist
        return nearest_node
    
    def get_nearby_nodes(self, new_node):
        nearby_nodes = []
        for node in self.tree:
            dist = np.linalg.norm(node.coordinates - new_node.coordinates)
            if dist < self.search_radius:
                nearby_nodes.append(node)
        return nearby_nodes

    def steer(self, from_node, to_node):
        direction = to_node.coordinates - from_node.coordinates
        dist = np.linalg.norm(direction)
        direction = direction / dist
        new_coordinates = from_node.coordinates + self.step_size * direction
        old_coordinates = from_node.coordinates
        
        col = collision_check(old_coordinates, new_coordinates, self.center, self.radius)
        
        if col.collision() == False:
            return None
    
        new_node = Node(new_coordinates)
        new_node.parent = from_node
        new_node.cost = from_node.cost + dist
        return new_node
    
    def rewire(self, new_node, nerby_nodes):
        for node in nerby_nodes:
            new_cost = new_node.cost + np.linalg.norm(node.coordinates - new_node.coordinates)
            if new_cost < node.cost:
                col = collision_check(node.coordinates, new_node.coordinates, self.center, self.radius)
                if col.collision() == False:
                    continue
                node.parent = new_node
                node.cost = new_cost

    def is_goal_reached(self, node):
        dist_to_goal = np.linalg.norm(node.coordinates - self.goal.coordinates)
        
        return dist_to_goal < self.step_size

    def generate_path(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            path.append(node.coordinates)
            node = node.parent
        return np.array(path[::-1])

    def plan(self):
        for _ in range(self.max_iter):
            random_node = self.get_random_node()
            nearest_node = self.get_nearest_node(random_node)
            new_node = self.steer(nearest_node, random_node)
            
            if new_node is None:  
                continue

            nearby_nodes = self.get_nearby_nodes(new_node)
            best_parent = nearest_node
            min_cost = nearest_node.cost + np.linalg.norm(nearest_node.coordinates - new_node.coordinates)
            for node in nearby_nodes:
                cost = node.cost + np.linalg.norm(node.coordinates - new_node.coordinates)
                if cost < min_cost:
                    best_parent = node
                    min_cost = cost

            new_node.parent = best_parent
            new_node.cost = min_cost
            self.tree.append(new_node)
            self.rewire(new_node, nearby_nodes)

            if self.is_goal_reached(new_node):
                print("Goal reached!")
                return self.generate_path(new_node)

        print("Goal not reached within max iterations")
        return None

    def visualize(self, path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])
    
    # Plot the tree
        for node in self.tree:
            if node.parent:
                ax.plot([node.coordinates[0], node.parent.coordinates[0]], 
                    [node.coordinates[1], node.parent.coordinates[1]], 
                    [node.coordinates[2], node.parent.coordinates[2]], 'g-')
    
    # Plot the path
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-', linewidth=2)
    
    # Plot start and goal
        ax.scatter(self.start.coordinates[0], self.start.coordinates[1], self.start.coordinates[2], c='b', s=50, label='Start')
        ax.scatter(self.goal.coordinates[0], self.goal.coordinates[1], self.goal.coordinates[2], c='r', s=50, label='Goal')
    
        ax.set_xlim(self.map_ranges[0][0], self.map_ranges[0][1])
        ax.set_ylim(self.map_ranges[1][0], self.map_ranges[1][1])
        ax.set_zlim(self.map_ranges[2][0], self.map_ranges[2][1])
    
    # Plot multiple obstacles (spheres)
        for i, center in enumerate(self.center):
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = self.radius[i] * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = self.radius[i] * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = self.radius[i] * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            ax.plot_surface(x, y, z, color='r', alpha=0.5)
    
        plt.legend()
        plt.show()



class Trajectory_Optimize:
    def __init__(self, path, centers, radius):
        self.path = path
        self.centers = centers  # List of obstacle centers
        self.radius = radius      # List of obstacle radii
    
    def pruning(self):
        q1 = self.path
        q2 = [q1[0]]
        qtemp = q1[0]
        
        for i in range(2, q1.shape[0]):
            # Check for collision along the path
            if collision_check(qtemp, q1[i], self.centers, self.radius).collision() == False:
                q2.append(q1[i - 1])
                qtemp = q1[i - 1]
                
        q2.append(q1[-1])
        q2 = np.array(q2)
        
        # Purning the path using spline interpolation
        tck, _ = splprep([q2[:, 0], q2[:, 1], q2[:, 2]], s=0, k=2)
        u_interp = np.linspace(0, 1, 30)
        x_interp, y_interp, z_interp = splev(u_interp, tck)
        path_interp = np.column_stack((x_interp, y_interp, z_interp))
        
        return q2, path_interp


def main():
    start = [ 50.06973358, -43.89813244,  70.85376038]
    goal = [-58.326769,   -52.94270309,  14.11190423]
    map_ranges = [(-90, 90), (-90, 90), (-90, 90)]
    center = [(0, -40, 50), (0, 40, 50)]
    radius = [20, 20]
    rrt_star = RRTStar(start, goal, map_ranges, center, radius, step_size = 1)
    path = rrt_star.plan()
    path, path_interp = Trajectory_Optimize(path, center, radius).pruning()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制路径
    ax.plot(path_interp[:, 0], path_interp[:, 1], path_interp[:, 2], label='Interpolated Path')

    # 绘制起点和终点
    ax.scatter(start[0], start[1], start[2], color='green', marker='o', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], color='red', marker='x', s=100, label='Goal')

    # 如果需要，可以绘制障碍物的中心点
    for c in center:
        ax.scatter(c[0], c[1], c[2], color='blue', marker='^', s=100, label='Obstacle Center')

    # 设置图例
    ax.legend()

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置坐标轴范围
    ax.set_xlim(map_ranges[0])
    ax.set_ylim(map_ranges[1])
    ax.set_zlim(map_ranges[2])

    # 显示图形
    plt.show()

if __name__ == '__main__':
    main()


