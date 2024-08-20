from itertools import combinations
import math
import numpy as np
import matplotlib.pyplot as plt
from complexe import Complex
import gudhi as gd

class Cech_Complex(Complex):
    def __init__(self, points=None):
        super().__init__(points)

    def filter_simplices(self):
        """
        Filter simplices with respect to the cech filtration up to dimension 2.
        """
        cech_skeleton = gd.RipsComplex(points=self.points)
        self.simplextree = cech_skeleton.create_simplex_tree(max_dimension=1)

        for (A, B, C) in combinations(range(len(self.points)), 3):
            circumcenter, circumradius = self.circumcenter_and_radius(self.points[A], self.points[B], self.points[C])
            if self.is_point_in_triangle(circumcenter, self.points[A], self.points[B], self.points[C]):
                filtration = 2*circumradius
            else:
                filtration = max(self.simplextree.filtration([A,B]), self.simplextree.filtration([B,C]), self.simplextree.filtration([A,C]))
            self.simplextree.insert([A, B, C], filtration)

        self.simplextree.compute_persistence(persistence_dim_max=True)

    def circumradius(self, A, B, C):
        # Compute the side lengths a, b, c
        a = np.linalg.norm(B - C)
        b = np.linalg.norm(A - C)
        c = np.linalg.norm(A - B)
        
        # Compute the semi-perimeter
        s = (a + b + c) / 2
        
        # Compute the area using Heron's formula
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        
        # Compute the circumradius
        R = (a * b * c) / (4 * area)
        
        return R
    
    def circumcenter_and_radius(self, A, B, C):
        # Unpack points
        x1, y1 = A
        x2, y2 = B
        x3, y3 = C
        
        # Calculate the determinant D
        D = 2 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        
        # Calculate circumcenter coordinates (x, y)
        x = ((x1**2 + y1**2)*(y2 - y3) + (x2**2 + y2**2)*(y3 - y1) + (x3**2 + y3**2)*(y1 - y2)) / D
        y = ((x1**2 + y1**2)*(x3 - x2) + (x2**2 + y2**2)*(x1 - x3) + (x3**2 + y3**2)*(x2 - x1)) / D
        
        # Circumcenter
        circumcenter = np.array([x, y])
        
        # Calculate the circumradius (distance to any vertex)
        radius = np.sqrt((x - x1)**2 + (y - y1)**2)
        
        return circumcenter, radius
    
    def is_point_in_triangle(self, pt, v1, v2, v3):
        """
        Check if the point pt is inside the triangle formed by points v1, v2, and v3.

        Parameters:
        - pt: Tuple of the point to check (x, y).
        - v1, v2, v3: Tuples of the triangle's vertices (x, y).

        Returns:
        - True if pt is inside the triangle, False otherwise.
        """

        # Convert points to numpy arrays for easier calculation
        pt = np.array(pt)
        v1 = np.array(v1)
        v2 = np.array(v2)
        v3 = np.array(v3)

        # Compute vectors
        v2_v1 = v2 - v1
        v3_v1 = v3 - v1
        pt_v1 = pt - v1

        # Compute dot products
        dot00 = np.dot(v3_v1, v3_v1)
        dot01 = np.dot(v3_v1, v2_v1)
        dot02 = np.dot(v3_v1, pt_v1)
        dot11 = np.dot(v2_v1, v2_v1)
        dot12 = np.dot(v2_v1, pt_v1)

        # Compute barycentric coordinates
        denom = dot00 * dot11 - dot01 * dot01
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom

        # Check if point is inside the triangle
        return (u >= 0) and (v >= 0) and (u + v <= 1)