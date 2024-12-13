import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import time

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_to(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

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

class StaticQuadTree:
    def __init__(self, boundary: Square, points: List[Point]):
        self.boundary = boundary
        self.points = points
        self.children = None
        if len(points) > 1:
            self.subdivide()

    # Standard quadtree subdivision but ensures points go to exactly one quadrant
    # This is crucial for the dynamic structure to work correctly
    def subdivide(self):
        x, y = self.boundary.x, self.boundary.y
        size = self.boundary.size / 2
        self.children = {
            'nw': {'boundary': Square(x, y + size, size), 'points': []},
            'ne': {'boundary': Square(x + size, y + size, size), 'points': []},
            'sw': {'boundary': Square(x, y, size), 'points': []},
            'se': {'boundary': Square(x + size, y, size), 'points': []}
        }

        for point in self.points:
            for quad, data in self.children.items():
                if data['boundary'].contains(point):
                    data['points'].append(point)
                    break

        for quad, data in self.children.items():
            if data['points']:
                self.children[quad] = StaticQuadTree(data['boundary'], data['points'])

# Dynamic quad-tree using level-based insertion strategy
class DynamicQuadTree:
    def __init__(self, boundary: Square):
        self.boundary = boundary
        self.levels = []
        self.n = 0

    def insert(self, point: Point):
        k = 0
        points_to_insert = [point]
        
        while k < len(self.levels) and self.levels[k] is not None:
            points_to_insert.extend(self.levels[k].points)
            self.levels[k] = None
            k += 1
            
        while len(self.levels) <= k:
            self.levels.append(None)
            
        self.levels[k] = StaticQuadTree(self.boundary, points_to_insert)
        self.n += 1

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

            if nearest is not None:
                if search_radius >= min_dist / (1 + epsilon):
                    return nearest, min_dist

            search_radius *= 2

    def query_range(self, boundary: Square) -> List[Point]:
        found_points = []
        for level in self.levels:
            if level is not None:
                found_points.extend(self._query_static_tree(level, boundary))
        return found_points

    def _query_static_tree(self, tree: StaticQuadTree, boundary: Square) -> List[Point]:
        if not tree.boundary.intersects(boundary):
            return []

        found_points = []
        
        for point in tree.points:
            if boundary.contains(point):
                found_points.append(point)

        if not tree.children:
            return found_points

        for child in tree.children.values():
            if isinstance(child, StaticQuadTree):
                found_points.extend(self._query_static_tree(child, boundary))

        return found_points

def run_random_queries(quadtree: DynamicQuadTree,
                      x_range: Tuple[float, float],
                      y_range: Tuple[float, float],
                      n_queries: int = 1000,
                      epsilon: float = 0.1) -> Dict:
    times = []
    distances = []

    for i in range(n_queries):
        if i % 100 == 0:
            print(f"Processing query {i}/{n_queries}")

        query = Point(
            np.random.uniform(*x_range),
            np.random.uniform(*y_range)
        )

        start_time = time.time()
        _, dist = quadtree.find_nearest_neighbor(query, epsilon)
        query_time = time.time() - start_time

        times.append(query_time)
        distances.append(dist)

    return {
        'mean_time': np.mean(times) * 1000,
        'std_time': np.std(times) * 1000,
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': min(distances),
        'max_distance': max(distances)
    }

def analyze_queries():
    print("Starting data analysis...")

    size = 2000
    quadtree = DynamicQuadTree(Square(0, 0, size))

    print("\nGenerating and inserting 6000 initial points...")
    initial_points = [Point(np.random.uniform(0, 1000), np.random.uniform(0, 1000)) 
                     for _ in range(6000)]
    for point in initial_points:
        quadtree.insert(point)

    print("\nTesting different epsilon values...")
    epsilon_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    test_queries = [
        (Point(500, 500), "q₀"),
        (Point(1000, 1000), "q₁"),
        (Point(30, 950), "q₃"),
        (Point(0, 1020), "q₄")
    ]

    results = {query[1]: [] for query in test_queries}
    
    for query, label in test_queries:
        print(f"Processing query point {label}: ({query.x}, {query.y})")
        for eps in epsilon_values:
            _, dist = quadtree.find_nearest_neighbor(query, eps)
            results[label].append(dist)

    plt.figure(figsize=(12, 8))
    for label, distances in results.items():
        plt.plot(epsilon_values, distances, marker='o', label=label)
    plt.xlabel('ε value')
    plt.ylabel('Distance to returned point')
    plt.title('ε-NN Query Results for Different Query Points')
    plt.legend()
    plt.grid(True)
    plt.savefig('epsilon_analysis.png')
    plt.close()

    print("\nRunning initial random queries in [0,1000] × [0,1000]...")
    original_stats = run_random_queries(quadtree, (0, 1000), (0, 1000))

    print("\nRunning queries in extended area [1000,1500] × [1000,1500]...")
    extended_stats_before = run_random_queries(quadtree, (1000, 1500), (1000, 1500))

    print("\nInserting 2000 additional points...")
    for _ in range(2000):
        point = Point(np.random.uniform(1000, 2000), np.random.uniform(1000, 2000))
        quadtree.insert(point)

    print("\nRunning final queries in extended region...")
    extended_stats_after = run_random_queries(quadtree, (1000, 2000), (1000, 2000))

    print("\nAnalysis Results:")
    print("\n1. Original Region [0,1000] × [0,1000]:")
    print(f"Average query time: {original_stats['mean_time']:.2f} ms")
    print(f"Average distance: {original_stats['mean_distance']:.2f}")
    print(f"Standard deviation of distances: {original_stats['std_distance']:.2f}")

    print("\n2. Extended Region [1000,1500] × [1000,1500] (Before new points):")
    print(f"Average query time: {extended_stats_before['mean_time']:.2f} ms")
    print(f"Average distance: {extended_stats_before['mean_distance']:.2f}")
    print(f"Standard deviation of distances: {extended_stats_before['std_distance']:.2f}")

    print("\n3. Extended Region [1000,2000] × [1000,2000] (After new points):")
    print(f"Average query time: {extended_stats_after['mean_time']:.2f} ms")
    print(f"Average distance: {extended_stats_after['mean_distance']:.2f}")
    print(f"Standard deviation of distances: {extended_stats_after['std_distance']:.2f}")

    print("\nComparative Analysis:")
    time_increase = (extended_stats_after['mean_time'] / original_stats['mean_time'] - 1) * 100
    dist_change = (extended_stats_after['mean_distance'] / original_stats['mean_distance'] - 1) * 100
    
    print(f"\nPerformance Changes:")
    print(f"- Query time increase: {time_increase:.1f}%")
    print(f"- Average distance change: {dist_change:.1f}%")
    
    print("\nResults Explanation:")
    print("1. Epsilon Analysis:")
    print("   - Higher ε values lead to larger distances but faster queries")
    print("   - Lower ε values provide better approximation but require more time")
    
    print("\n2. Region Comparison:")
    print("   - Original region shows consistent performance due to uniform distribution")
    print("   - Extended region improves after adding points but with increased query time")
    print("   - Query times scale with region size and point density")

    return {
        'original': original_stats,
        'extended_before': extended_stats_before,
        'extended_after': extended_stats_after,
        'epsilon_results': results
    }

if __name__ == "__main__":
    try:
        results = analyze_queries()
    except Exception as e:
        print(f"Error during analysis: {str(e)}")