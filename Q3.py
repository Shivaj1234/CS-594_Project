import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Optional
import random

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_to(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    # Required for deletion: points must be comparable with proper floating-point tolerance
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10

    def __hash__(self):
        return hash((self.x, self.y))

class Square:
    def __init__(self, x: float, y: float, size: float):
        self.x = x
        self.y = y
        self.size = size

    def contains(self, point: Point) -> bool:
        return (self.x <= point.x <= self.x + self.size and
                self.y <= point.y <= self.y + self.size)

    def intersects(self, other: 'Square') -> bool:
        return not (other.x > self.x + self.size or
                   other.x + other.size < self.x or
                   other.y > self.y + self.size or
                   other.y + other.size < self.y)

class DQuadNode:
  # Enhanced node for deletion support: tracks representatives and node activity
    def __init__(self, boundary: Square, parent=None):
        self.boundary = boundary
        self.points = []
        self.children = {'nw': None, 'ne': None, 'sw': None, 'se': None}
        self.divided = False
        self.representative = None   # Each node maintains a representative point
        self.is_active = True        # Nodes can be marked inactive after deletion
        self.parent = parent         # Parent reference for updating representatives
        
    # Chooses first available point as representative
    # Returns False if no points available, marking node as potentially inactive
    def select_representative(self):
        if self.points:
            self.representative = self.points[0]
            return True
        return False

class DQuadTree:
    def __init__(self, boundary: Square, points: List[Point] = None):
        self.root = DQuadNode(boundary)
        self.initial_count = len(points) if points else 0
        self.deletion_count = 0

        if points:
            for point in points:
                self._insert_point(point)

    def _insert_point(self, point: Point):
        node = self.root

        while True:
            node.points.append(point)
            if not node.representative:
                node.representative = point

            if not node.divided and len(node.points) > 1:
                self._subdivide(node)

            if not node.divided:
                break

            found_child = False
            for child in node.children.values():
                if child.boundary.contains(point):
                    node = child
                    found_child = True
                    break

            if not found_child:
                break

    def _subdivide(self, node: DQuadNode):
        x = node.boundary.x
        y = node.boundary.y
        size = node.boundary.size / 2

        node.children = {
            'nw': DQuadNode(Square(x, y + size, size), node),
            'ne': DQuadNode(Square(x + size, y + size, size), node),
            'sw': DQuadNode(Square(x, y, size), node),
            'se': DQuadNode(Square(x + size, y, size), node)
        }
        node.divided = True

        points = node.points.copy()
        node.points = []

        for point in points:
            for child in node.children.values():
                if child.boundary.contains(point):
                    child.points.append(point)
                    if not child.representative:
                        child.representative = point

    def delete_point(self, point: Point) -> Tuple[bool, float]:
        start_time = time.time()

        if self.deletion_count >= self.initial_count // 2:
            old_count = self.count_points()
            remaining_points = self._collect_unique_points()
            self.__init__(self.root.boundary, remaining_points)
            new_count = self.count_points()
            print(f"Reconstruction: {old_count} points before, {new_count} points after")
            return self.delete_point(point)

        leaf_node = self._find_leaf_node(self.root, point)
        if not leaf_node or point not in leaf_node.points:
            return False, time.time() - start_time

        leaf_node.points.remove(point)

        # Update representatives from leaf to root
        # If a node loses its representative, try to select new one
        # If no points available, mark node as inactive
        current = leaf_node
        while current:
            if current.representative == point:
                if not current.select_representative():
                    current.is_active = False
            current = current.parent

        self.deletion_count += 1
        return True, time.time() - start_time

    def _find_leaf_node(self, node: DQuadNode, point: Point) -> Optional[DQuadNode]:
        if not node.boundary.contains(point):
            return None

        if not node.divided:
            return node

        for child in node.children.values():
            if child.boundary.contains(point):
                result = self._find_leaf_node(child, point)
                if result:
                    return result
        return None

    def _collect_unique_points(self) -> List[Point]:
        points_set = set()

        def collect_from_leaves(node):
            if not node.divided:
                points_set.update(node.points)
            else:
                for child in node.children.values():
                    if child:
                        collect_from_leaves(child)

        collect_from_leaves(self.root)
        return list(points_set)

    def find_nearest_neighbor(self, query_point: Point, epsilon: float = 0.1) -> Tuple[Point, float]:
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")

        search_radius = 1.0
        nearest = None
        min_dist = float('inf')

        while True:
            search_box = Square(
                query_point.x - search_radius,
                query_point.y - search_radius,
                2 * search_radius
            )

            candidates = self.query_range(search_box)
            for point in candidates:
                dist = point.distance_to(query_point)
                if dist < min_dist:
                    min_dist = dist
                    nearest = point

            if nearest and search_radius >= min_dist / (1 + epsilon):
                return nearest, min_dist

            search_radius *= 2

    def query_range(self, boundary: Square) -> List[Point]:
        def query_node(node):
            if not node.is_active or not node.boundary.intersects(boundary):
                return []

            found_points = []
            if not node.divided:
                found_points.extend(p for p in node.points if boundary.contains(p))
            else:
                for child in node.children.values():
                    found_points.extend(query_node(child))

            return found_points

        return query_node(self.root)

    def count_points(self) -> int:
        return len(self._collect_unique_points())

    def delete_points_in_box(self, box: Square) -> Tuple[List[Point], float]:
        points = list(set(self.query_range(box)))
        times = []
        deleted = []

        for point in points:
            success, time_taken = self.delete_point(point)
            if success:
                deleted.append(point)
                times.append(time_taken)

        return deleted, np.mean(times) if times else 0

def run_experiments():
    points = []
    with open('dataset.txt', 'r', encoding='utf-8-sig') as f:
        for line in f:
            try:
                x, y = map(float, line.strip().split(','))
                points.append(Point(x, y))
            except ValueError as e:
                print(f"Skipping invalid line: {line.strip()} - {str(e)}")
                continue

    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)
    size = max(max_x - min_x, max_y - min_y)
    margin = size * 0.05
    boundary = Square(min_x - margin, min_y - margin, size + 2 * margin)

    tree = DQuadTree(boundary, points)
    print(f"Initial tree contains {tree.count_points()} points")

    box1 = Square(450, 450, 100)
    deleted1, avg_time1 = tree.delete_points_in_box(box1)
    print(f"\nDeleted {len(deleted1)} points from first box")
    print(f"Average deletion time: {avg_time1*1000:.2f} ms")

    epsilon_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    q0 = Point(500, 500)
    q0_distances = []

    for eps in epsilon_values:
        _, dist = tree.find_nearest_neighbor(q0, eps)
        q0_distances.append(dist)

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, q0_distances, 'b-o', label='q₀ = (500,500)')
    plt.xlabel('ε value')
    plt.ylabel('Distance to returned point')
    plt.title('ε-NN Query Results for q₀ after First Box Deletion')
    plt.grid(True)
    plt.legend()
    plt.show()

    box2 = Square(900, 900, 100)
    deleted2, avg_time2 = tree.delete_points_in_box(box2)
    print(f"\nDeleted {len(deleted2)} points from second box")
    print(f"Average deletion time: {avg_time2*1000:.2f} ms")

    q1 = Point(1000, 1000)
    q1_distances = []

    for eps in epsilon_values:
        _, dist = tree.find_nearest_neighbor(q1, eps)
        q1_distances.append(dist)

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, q1_distances, 'r-o', label='q₁ = (1000,1000)')
    plt.xlabel('ε value')
    plt.ylabel('Distance to returned point')
    plt.title('ε-NN Query Results for q₁ after Second Box Deletion')
    plt.grid(True)
    plt.legend()
    plt.show()

    points_needed = (tree.initial_count // 2) + 1 - tree.deletion_count
    print(f"\nDeleting {points_needed} more points to trigger reconstruction...")
    remaining = tree._collect_unique_points()
    to_delete = random.sample(remaining, min(points_needed, len(remaining)))

    deletion_times = []
    for point in to_delete:
        _, time_taken = tree.delete_point(point)
        deletion_times.append(time_taken)

    print(f"Average deletion time before reconstruction: {np.mean(deletion_times)*1000:.2f} ms")

    remaining = tree._collect_unique_points()
    to_delete = random.sample(remaining, min(1000, len(remaining)))

    deletion_times = []
    for point in to_delete:
        _, time_taken = tree.delete_point(point)
        deletion_times.append(time_taken)

    print(f"\nAverage deletion time after reconstruction: {np.mean(deletion_times)*1000:.2f} ms")

    q1_distances = []
    q2 = Point(30, 950)
    q2_distances = []

    for eps in epsilon_values:
        _, dist1 = tree.find_nearest_neighbor(q1, eps)
        _, dist2 = tree.find_nearest_neighbor(q2, eps)
        q1_distances.append(dist1)
        q2_distances.append(dist2)

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, q1_distances, 'r-o', label='q₁ = (1000,1000)')
    plt.plot(epsilon_values, q2_distances, 'g-o', label='q₂ = (30,950)')
    plt.xlabel('ε value')
    plt.ylabel('Distance to returned point')
    plt.title('ε-NN Query Results after Reconstruction')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"\nPoints still stored in data structure: {tree.count_points()}")

    print("\nRunning 1000 random queries in [0, 1000] × [0, 1000]...")
    times1 = []
    distances1 = []

    for _ in range(1000):
        query = Point(
            np.random.uniform(0, 1000),
            np.random.uniform(0, 1000)
        )
        start_time = time.time()
        _, dist = tree.find_nearest_neighbor(query, 0.1)
        times1.append(time.time() - start_time)
        distances1.append(dist)

    print(f"Average query time: {np.mean(times1)*1000:.2f} ms")
    print(f"Average distance: {np.mean(distances1):.2f}")

    print("\nRunning 1000 random queries in [1000, 1500] × [1000, 1500]...")
    times2 = []
    distances2 = []

    for _ in range(1000):
        query = Point(
            np.random.uniform(1000, 1500),
            np.random.uniform(1000, 1500)
        )
        start_time = time.time()
        _, dist = tree.find_nearest_neighbor(query, 0.1)
        times2.append(time.time() - start_time)
        distances2.append(dist)

    print(f"Average query time: {np.mean(times2)*1000:.2f} ms")
    print(f"Average distance: {np.mean(distances2):.2f}")

if __name__ == "__main__":
    run_experiments()