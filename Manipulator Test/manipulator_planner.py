
import numpy as np
import manipulator_collision

class Node:
    def __init__(self, q, parent=None):
        self.q = q
        self.parent = parent

class RRTPlanner:
    def __init__(self, model, data, step_size=0.2, max_iter=2000, goal_bias=0.1):
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.checker = manipulator_collision.CollisionChecker(model, data)
        # Monkey patch simple edge checker if not exists
        if not hasattr(self.checker, 'check_edge'):
             def check_edge(q1, q2, steps=5):
                 for alpha in np.linspace(0, 1, steps):
                     q = q1 * (1-alpha) + q2 * alpha
                     if self.checker.is_collision(q): return True
                 return False
             self.checker.check_edge = check_edge
             
        self.joint_limits = np.array([[-3.14, 3.14]] * 6) # Approximate
        
        # Read actual limits if available
        if model.jnt_limited is not None:
             # Logic to read limits map
             pass

    def plan(self, q_start, q_goal):
        """
        RRT Plan from q_start to q_goal.
        Returns path (list of q arrays) or None.
        """
        print(f"DEBUG: Planning from {q_start} to {q_goal}")
        
        # Check start/goal validity
        if self.checker.is_collision(q_start):
            print("DEBUG: Start configuration is in collision!")
            return None
        if self.checker.is_collision(q_goal):
            print("DEBUG: Goal configuration is in collision!")
            return None
            
        tree = [Node(q_start)]
        
        for i in range(self.max_iter):
            # 1. Sample
            if np.random.rand() < self.goal_bias:
                q_rand = q_goal
            else:
                q_rand = np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])
                
            # 2. Nearest
            node_nearest = min(tree, key=lambda n: np.linalg.norm(n.q - q_rand))
            q_near = node_nearest.q
            
            # 3. Steer
            direction = q_rand - q_near
            dist = np.linalg.norm(direction)
            
            if dist > self.step_size:
                q_new = q_near + (direction / dist) * self.step_size
            else:
                q_new = q_rand
                
            # 4. Collision Check (Point and Edge)
            # Check Point
            if not self.checker.is_collision(q_new):
                # Check Edge (Interpolate)
                # Simple check: midpoint
                mid = (q_near + q_new) / 2
                if not self.checker.is_collision(mid):
                    # Add Node
                    new_node = Node(q_new, node_nearest)
                    tree.append(new_node)
                    
                    # Check Goal
                    if np.linalg.norm(q_new - q_goal) < self.step_size:
                        print(f"DEBUG: Goal reached at iter {i}")
                        return self.extract_path(new_node, q_goal)
        
        print("DEBUG: RRT Max iterations reached. No path found.")
        return None

        return None

    def extract_path(self, end_node, q_goal):
        path = [q_goal]
        curr = end_node
        while curr is not None:
            path.append(curr.q)
            curr = curr.parent
        return path[::-1] # Reverse

class RRTStarPlanner(RRTPlanner):
    def __init__(self, model, data, step_size=0.2, max_iter=2000, goal_bias=0.1, connect_circle_dist=0.5):
        super().__init__(model, data, step_size, max_iter, goal_bias)
        self.connect_circle_dist = connect_circle_dist
        
    def plan(self, q_start, q_goal):
        print(f"DEBUG: RRT* Planning from {q_start} to {q_goal}")
        
        if self.checker.is_collision(q_start) or self.checker.is_collision(q_goal):
            print("DEBUG: Start or Goal in collision!")
            return None
            
        # RRT* Node needs 'cost' attribute
        start_node = Node(q_start)
        start_node.cost = 0.0
        tree = [start_node]
        
        best_goal_node = None
        min_goal_dist = float('inf')
        
        for i in range(self.max_iter):
            # 1. Sample
            if np.random.rand() < self.goal_bias:
                q_rand = q_goal
            else:
                q_rand = np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])
                
            # 2. Nearest
            node_nearest = min(tree, key=lambda n: np.linalg.norm(n.q - q_rand))
            q_near = node_nearest.q
            
            # 3. Steer
            direction = q_rand - q_near
            dist = np.linalg.norm(direction)
            if dist > self.step_size:
                q_new = q_near + (direction / dist) * self.step_size
            else:
                q_new = q_rand
                
            if self.checker.is_collision(q_new):
                continue
                
            # 4. Choose Parent (Optimization Step 1)
            # Find all nodes within radius
            near_nodes = [n for n in tree if np.linalg.norm(n.q - q_new) < self.connect_circle_dist]
            
            node_min = node_nearest
            c_min = node_nearest.cost + np.linalg.norm(q_new - node_nearest.q)
            
            for node_near in near_nodes:
                c_new = node_near.cost + np.linalg.norm(q_new - node_near.q)
                if c_new < c_min:
                    # Check collision for potential new parent
                    if not self.checker.check_edge(node_near.q, q_new): # Assuming check_edge exists or we implement it
                        node_min = node_near
                        c_min = c_new
                        
            # Add Node
            new_node = Node(q_new, node_min)
            new_node.cost = c_min
            tree.append(new_node)
            
            # 5. Rewire (Optimization Step 2)
            # Check if new node can be a better parent for neighbors
            for node_near in near_nodes:
                c_improve = new_node.cost + np.linalg.norm(node_near.q - new_node.q)
                if c_improve < node_near.cost:
                     if not self.checker.check_edge(new_node.q, node_near.q):
                         node_near.parent = new_node
                         node_near.cost = c_improve
                         
            # Check Goal
            d_goal = np.linalg.norm(q_new - q_goal)
            if d_goal < self.step_size:
                if best_goal_node is None or new_node.cost < best_goal_node.cost:
                    best_goal_node = new_node
                    print(f"DEBUG: RRT* Found path cost {new_node.cost:.2f} at iter {i}")
                    
        if best_goal_node:
            return self.extract_path(best_goal_node, q_goal)
        else:
            print("DEBUG: RRT* Failed")
            return None

def smooth_path_bspline(path, s=0.0):
    """
    Smooth path using B-Spline.
    path: List of arrays or numpy array (N, 6)
    """
    import scipy.interpolate
    try:
        path = np.array(path)
        if len(path) < 3: return path
        
        # Remove duplicates
        diff = np.linalg.norm(np.diff(path, axis=0), axis=1)
        # Keep indices where diff > epsilon (plus first point)
        keep = np.concatenate([[True], diff > 1e-3])
        path = path[keep]
        
        if len(path) < 3: return path
        
        # Spline
        tck, u = scipy.interpolate.splprep(path.T, s=s, k=2) # k=2 ranges 3 points
        u_fine = np.linspace(0, 1, num=len(path)*5) # 5x resolution
        new_points = scipy.interpolate.splev(u_fine, tck)
        
        return np.array(new_points).T
    except Exception as e:
        print(f"Smoothing Failed: {e}")
        return path

        return np.array(new_points).T
    except Exception as e:
        print(f"Smoothing Failed: {e}")
        return path

class PRMPlanner(RRTPlanner):
    def __init__(self, model, data, n_samples=200, k_neighbors=10):
        super().__init__(model, data)
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
        self.roadmap = [] # List of nodes
        self.adj = {} # Dict of adjacency: node_idx -> list of (neighbor_idx, cost)
        self.initialized = False
        
    def build_roadmap(self):
        print(f"PRM: Building Roadmap with {self.n_samples} samples...")
        self.roadmap = []
        self.adj = {}
        
        # 1. Sample Free Space
        while len(self.roadmap) < self.n_samples:
            q = np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])
            if not self.checker.is_collision(q):
                self.roadmap.append(q)
                
        # 2. Connect Neighbors
        for i in range(self.n_samples):
            self.adj[i] = []
            # Find K Nearest
            dists = []
            for j in range(self.n_samples):
                if i == j: continue
                d = np.linalg.norm(self.roadmap[i] - self.roadmap[j])
                dists.append((d, j))
            
            dists.sort()
            neighbors = dists[:self.k_neighbors]
            
            for d, j in neighbors:
                # Check Edge
                if not self.checker.check_edge(self.roadmap[i], self.roadmap[j]):
                    self.adj[i].append((j, d))
                    
        print(f"PRM: Roadmap built with {self.n_samples} nodes.")
        self.initialized = True
        
    def plan(self, q_start, q_goal):
        if not self.initialized:
            self.build_roadmap()
            
        print(f"PRM: Planning...")
        
        # Add Start/Goal to roadmap temporarily checking collision
        if self.checker.is_collision(q_start) or self.checker.is_collision(q_goal):
            print("PRM: Start/Goal in Collision")
            return None

        # Helper to find nearest connection to roadmap
        def connect_to_roadmap(q):
            best_idx = -1
            best_dist = float('inf')
            
            for i, node in enumerate(self.roadmap):
                 d = np.linalg.norm(q - node)
                 if d < best_dist:
                     if not self.checker.check_edge(q, node):
                         best_dist = d
                         best_idx = i
            return best_idx, best_dist

        start_idx, d_start = connect_to_roadmap(q_start)
        goal_idx, d_goal = connect_to_roadmap(q_goal)
        
        if start_idx == -1 or goal_idx == -1:
            print("PRM: Could not connect Start/Goal to roadmap.")
            return None
            
        # Search (Dijkstra/A*)
        # Open set: (cost, current_idx, parent_idx)
        import heapq
        queue = [(0, start_idx, -1)]
        visited = {} # idx -> parent_idx
        costs = {start_idx: 0}
        
        path_found = False
        
        while queue:
            c, curr, parent = heapq.heappop(queue)
            
            if curr in visited: continue
            visited[curr] = parent
            
            if curr == goal_idx:
                path_found = True
                break
                
            for neighbor, weight in self.adj.get(curr, []):
                new_cost = c + weight
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor, curr))
                    
        if path_found:
            # Reconstruct
            path_indices = []
            curr = goal_idx
            while curr != -1:
                path_indices.append(curr)
                curr = visited[curr]
            path_indices = path_indices[::-1] # Reverse
            
            # Map back to configs
            path = [q_start] + [self.roadmap[i] for i in path_indices] + [q_goal]
            return path
        else:
            print("PRM: No path found in graph.")
            return None
