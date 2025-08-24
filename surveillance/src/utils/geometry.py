# FILE: src/utils/geometry.py

"""
Geometric utility functions.
"""

import numpy as np
from typing import List, Tuple


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm.
    
    Args:
        point: (x, y) coordinates
        polygon: List of [x, y] polygon vertices
        
    Returns:
        True if point is inside polygon
    """
    if not polygon or len(polygon) < 3:
        return False
    
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1, point2: (x, y) coordinates
        
    Returns:
        Distance between points
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculate_polygon_area(polygon: List[List[float]]) -> float:
    """
    Calculate area of polygon using shoelace formula.
    
    Args:
        polygon: List of [x, y] vertices
        
    Returns:
        Polygon area
    """
    if len(polygon) < 3:
        return 0.0
    
    area = 0.0
    n = len(polygon)
    
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    
    return abs(area) / 2.0


def is_point_near_line(point: Tuple[float, float], line_start: Tuple[float, float],
                      line_end: Tuple[float, float], threshold: float = 5.0) -> bool:
    """
    Check if point is near a line segment.
    
    Args:
        point: Point to test
        line_start: Start of line segment
        line_end: End of line segment
        threshold: Maximum distance to consider "near"
        
    Returns:
        True if point is within threshold distance of line
    """
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Calculate distance from point to line segment
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:
        # Line segment is a point
        distance = calculate_distance(point, line_start)
    else:
        param = dot / len_sq
        
        if param < 0:
            # Closest point is line_start
            distance = calculate_distance(point, line_start)
        elif param > 1:
            # Closest point is line_end
            distance = calculate_distance(point, line_end)
        else:
            # Closest point is on the line segment
            closest_x = x1 + param * C
            closest_y = y1 + param * D
            distance = calculate_distance(point, (closest_x, closest_y))
    
    return distance <= threshold


def calculate_angle_between_points(center: Tuple[float, float], 
                                  point1: Tuple[float, float],
                                  point2: Tuple[float, float]) -> float:
    """
    Calculate angle between three points (in radians).
    
    Args:
        center: Center point (vertex of angle)
        point1: First point
        point2: Second point
        
    Returns:
        Angle in radians (0 to pi)
    """
    cx, cy = center
    x1, y1 = point1
    x2, y2 = point2
    
    # Create vectors from center to each point
    v1 = (x1 - cx, y1 - cy)
    v2 = (x2 - cx, y2 - cy)
    
    # Calculate dot product and magnitudes
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Calculate angle using arccosine
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle floating point errors
    
    return np.arccos(cos_angle)


def create_bounding_box_from_points(points: List[Tuple[float, float]]) -> List[float]:
    """
    Create axis-aligned bounding box from list of points.
    
    Args:
        points: List of (x, y) coordinates
        
    Returns:
        Bounding box [x, y, w, h]
    """
    if not points:
        return [0, 0, 0, 0]
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def scale_polygon(polygon: List[List[float]], scale_x: float, scale_y: float) -> List[List[float]]:
    """
    Scale polygon by given factors.
    
    Args:
        polygon: List of [x, y] vertices
        scale_x: X scaling factor
        scale_y: Y scaling factor
        
    Returns:
        Scaled polygon
    """
    return [[x * scale_x, y * scale_y] for x, y in polygon]


def translate_polygon(polygon: List[List[float]], dx: float, dy: float) -> List[List[float]]:
    """
    Translate polygon by given offset.
    
    Args:
        polygon: List of [x, y] vertices
        dx: X offset
        dy: Y offset
        
    Returns:
        Translated polygon
    """
    return [[x + dx, y + dy] for x, y in polygon]
