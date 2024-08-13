from itertools import combinations
import math
import numpy as np
import matplotlib.pyplot as plt
from complexe import Complex
import gudhi as gd

class Cech_Complex(Complex):
    def __init__(self, initial_epsilon, points=None):
        super().__init__(initial_epsilon, points)

    def filter_one_simplices(self, eps:float):
        """
        Filter one simplices, with respect to the filtration parameter eps.

        Parameters:
        eps: float - filtration parameter
        """
        self.edges = np.array([
            (i, j)
            for i, j in combinations(range(len(self.points)), 2)
            if np.linalg.norm(self.points[i] - self.points[j]) < 2 * eps
        ])
    
    def _intersection_points(self, c1, c2, eps):
        # Calculate the distance between the two circle centers
        d = np.linalg.norm(c1-c2)
        
        # Check if circles intersect
        if d > 2 * eps or d == 0:
            return []  # No intersection or circles are coincident
        
        # Calculate the midpoint between the centers
        a = d / 2
        h = math.sqrt(eps**2 - a**2)
        
        # Midpoint between the two centers
        x3 = c1[0] + (c2[0] - c1[0]) * a / d
        y3 = c1[1] + (c2[1] - c1[1]) * a / d
        
        # Intersection points
        intersect1 = (x3 + h * (c2[1] - c1[1]) / d, y3 - h * (c2[0] - c1[0]) / d)
        intersect2 = (x3 - h * (c2[1] - c1[1]) / d, y3 + h * (c2[0] - c1[0]) / d)
        
        return [intersect1, intersect2]
    
    def _point_in_circle(self, point, circle, eps):
        return round(np.linalg.norm(point-circle), 1) <= eps
    
    def all_three_intersect(self, c1, c2, c3, eps):
        points12 = self._intersection_points(c1, c2, eps)
        points13 = self._intersection_points(c1, c3, eps)
        points23 = self._intersection_points(c2, c3, eps)

        for point in points12 + points13 + points23:
            if self._point_in_circle(point, c1, eps) and self._point_in_circle(point, c2, eps) and self._point_in_circle(point, c3, eps):
                return True
        return False
    
    def all_circles_intersect(self, points, R):
        center_of_mass = np.mean(points, axis=0)
        return all(np.linalg.norm(center_of_mass-p) < R for p in points)
    
    def filter_two_simplices(self, eps):
        """
        Filter two simplices, with respect to the filtration parameter eps.

        Parameters:
        eps: float - filtration parameter
        """
        self.triangles = np.array([
            (c1, c2, c3)
            for c1, c2, c3 in combinations(range(len(self.points)), 3)
                if all(
                np.linalg.norm(self.points[i] - self.points[j]) <= 2 * eps
                for i, j in combinations([c1, c2, c3], 2)
            )
            and self.all_three_intersect(self.points[c1], self.points[c2], self.points[c3], eps)
        ])

