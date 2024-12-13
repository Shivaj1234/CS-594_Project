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

    # Checking if this square overlaps with another square
    # This is critical for range queries and ε-NN search
    def intersects(self, other: 'Square') -> bool:
        return not (other.x > self.x + self.size or
                   other.x + other.size < self.x or
                   other.y > self.y + self.size or
                   other.y + other.size < self.y)

class QuadTree:
    def __init__(self, boundary: Square, points: List[Point] = None):
        self.boundary = boundary
        self.points = points if points else []
        self.children = {'nw': None, 'ne': None, 'sw': None, 'se': None}
        self.divided = False

        if len(self.points) > 1:
            self.subdivide()

    def subdivide(self):
        x = self.boundary.x
        y = self.boundary.y
        size = self.boundary.size / 2

        boundaries = {
            'nw': Square(x, y + size, size),
            'ne': Square(x + size, y + size, size),
            'sw': Square(x, y, size),
            'se': Square(x + size, y, size)
        }

        child_points = {k: [] for k in boundaries.keys()}
        for point in self.points:
            for quad, boundary in boundaries.items():
                if boundary.contains(point):
                    child_points[quad].append(point)

        for quad in boundaries.keys():
            self.children[quad] = QuadTree(boundaries[quad], child_points[quad])

        self.divided = True

    # Implementation for (1+ε)-approximate nearest neighbor search
    # Key algorithm steps:
    # 1. Starts with initial search radius
    # 2. Finds points within current search radius
    # 3. If found point satisfies (1+ε) approximation, returns it
    # 4. Otherwise, doubles the radius and repeats
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
        
        if not self.boundary.intersects(boundary):
            return []

        found_points = []

        
        for point in self.points:
            if boundary.contains(point):
                found_points.append(point)

        if not self.divided:
            return found_points

       
        for child in self.children.values():
            if child:
                found_points.extend(child.query_range(boundary))

        return found_points

def analyze_queries(data_file: str):
    print("Starting data analysis...")

    
    points = []
    try:
        with open(data_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                try:
                    x, y = map(float, line.strip().split(','))
                    points.append(Point(x, y))
                except ValueError as e:
                    print(f"Skipping invalid line: {line.strip()} - {str(e)}")
                    continue

        if not points:
            raise ValueError("No valid points were read from the file")

    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data file: {data_file}")

   
    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)

    print(f"\nData boundaries:")
    print(f"X range: [{min_x:.2f}, {max_x:.2f}]")
    print(f"Y range: [{min_y:.2f}, {max_y:.2f}]")
    print(f"Total points: {len(points)}")

    size = max(max_x - min_x, max_y - min_y)
    margin = size * 0.05
    boundary = Square(min_x - margin, min_y - margin, size + 2 * margin)
    quadtree = QuadTree(boundary, points)

    
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
    plt.title('ε-NN Query Results')
    plt.legend()
    plt.grid(True)
    plt.savefig('epsilon_analysis.png')
    plt.close()

  
    def run_random_queries(x_range: Tuple[float, float],
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

    print("\nRunning random queries in [0,1000] × [0,1000]...")
    region1_stats = run_random_queries((0, 1000), (0, 1000))

    print("\nRunning random queries in [1000,1500] × [1000,1500]...")
    region2_stats = run_random_queries((1000, 1500), (1000, 1500))

    print("\nResults for [0,1000] × [0,1000]:")
    for key, value in region1_stats.items():
        print(f"{key}: {value:.2f}")

    print("\nResults for [1000,1500] × [1000,1500]:")
    for key, value in region2_stats.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    try:
        analyze_queries('dataset.txt')
    except Exception as e:
        print(f"Error during analysis: {str(e)}")