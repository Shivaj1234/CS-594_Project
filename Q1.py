import math
import time
import numpy as np
from typing import List, Tuple


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

   
    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Square:
    def __init__(self, x: float, y: float, size: float):
        self.x = x
        self.y = y
        self.size = size

    def contains(self, point: Point) -> bool:
        return (self.x <= point.x <= self.x + self.size and
                self.y <= point.y <= self.y + self.size)

# Quad-tree implementation for spatial partitioning
class QuadTree:
    def __init__(self, boundary: Square, points: List[Point] = None):
        self.boundary = boundary
        self.points = points if points else []
        self.children = {'nw': None, 'ne': None, 'sw': None, 'se': None}
        self.divided = False

        if len(self.points) > 1:
            self.subdivide()

    # Key method that implements spatial partitioning
    def subdivide(self):
        x = self.boundary.x
        y = self.boundary.y
        size = self.boundary.size / 2

        # Create boundaries ensuring no gaps or overlaps
        # Each vertex (except edges) is shared by exactly 4 squares
        boundaries = {
            'nw': Square(x, y + size, size),        # Northwest: upper-left
            'ne': Square(x + size, y + size, size), # Northeast: upper-right
            'sw': Square(x, y, size),               # Southwest: lower-left
            'se': Square(x + size, y, size)         # Southeast: lower-right
        }

        child_points = {k: [] for k in boundaries.keys()}
        
        # Distribute points to quadrants
        for point in self.points:
            for quad, boundary in boundaries.items():
                if boundary.contains(point):
                    child_points[quad].append(point)

        for quad in boundaries.keys():
            self.children[quad] = QuadTree(boundaries[quad], child_points[quad])

        self.divided = True

    def get_height(self) -> int:
        if not self.divided:
            return 0
        return 1 + max(child.get_height() for child in self.children.values() if child)


def compute_spread(points: List[Point]) -> Tuple[float, float, float, Tuple[Point, Point], Tuple[Point, Point]]:
    max_dist = float('-inf')
    min_dist = float('inf')
    max_pair = None
    min_pair = None

    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = points[i].distance_to(points[j])
            if dist > 0:  # Crucial check for distinct points
                if dist > max_dist:
                    max_dist = dist
                    max_pair = (points[i], points[j])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (points[i], points[j])

    return max_dist/min_dist, max_dist, min_dist, max_pair, min_pair


def analyze_dataset(filename: str):
    points = []

    #
    print("Reading dataset...")
    with open(filename, 'r', encoding='utf-8-sig') as f:
        for line in f:
            try:
                x, y = map(float, line.strip().split(','))
                points.append(Point(x, y))
            except ValueError as e:
                print(f"Skipping invalid line: {line.strip()} - {str(e)}")
                continue

    print(f"Read {len(points)} points")

    # Calculating bounding box for all points
    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)
    size = max(max_x - min_x, max_y - min_y)

    # Adding margins for visualization
    margin = size * 0.05
    min_x -= margin
    min_y -= margin
    size += 2 * margin

    
    print("\nBuilding QuadTree...")
    start_time = time.time()
    quadtree = QuadTree(Square(min_x, min_y, size), points)
    construction_time = time.time() - start_time

    print("\nCalculating spread and height...")
    spread, max_dist, min_dist, max_pair, min_pair = compute_spread(points)
    height = quadtree.get_height()

    
    print("\nResults:")
    print(f"Construction time: {construction_time:.4f} seconds")
    print(f"\nSpread Analysis:")
    print(f"Maximum distance: {max_dist:.4f}")
    print(f"  Between points: ({max_pair[0].x:.2f}, {max_pair[0].y:.2f}) and "
          f"({max_pair[1].x:.2f}, {max_pair[1].y:.2f})")
    print(f"Minimum distance: {min_dist:.4f}")
    print(f"  Between points: ({min_pair[0].x:.2f}, {min_pair[0].y:.2f}) and "
          f"({min_pair[1].x:.2f}, {min_pair[1].y:.2f})")
    print(f"Spread (max/min ratio): {spread:.4f}")

    print(f"\nTree Analysis:")
    print(f"Tree Height: {height}")
    print(f"log₂(spread): {math.log2(spread):.4f}")
    print(f"Height/log₂(spread) ratio: {height/math.log2(spread):.4f}")

    
    print("\nRelationship between spread and height:")
    print("The height of the quadtree is closely related to the spread of the point set.")
    print("Theoretically, the height should be O(log(spread)) because:")
    print("1. Each level of the tree divides space into equal quadrants")
    print("2. The number of subdivisions needed to separate close points is proportional to log(spread)")
    print(f"In this case, height ({height}) is approximately {height/math.log2(spread):.2f} times log₂(spread) ({math.log2(spread):.2f})")

    return quadtree

if __name__ == "__main__":
    try:
        quadtree = analyze_dataset('dataset.txt')
    except Exception as e:
        print(f"Error: {str(e)}")