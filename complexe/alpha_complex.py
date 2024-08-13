from complexe import Complex
import numpy as np
from itertools import combinations
from scipy.spatial import Delaunay


class Alpha_Complex(Complex):
    def __init__(self, initial_epsilon, points=None):
        self.delauney_triangles = Delaunay(points).simplices
        self.delauney_edges = np.array([
            (i, j) 
            for triangle in self.delauney_triangles
            for i, j in combinations(triangle, 2)
        ])
        super().__init__(initial_epsilon, points)
    
    def filter_one_simplices(self, eps:float):
        """
        Filter one simplices, with respect to the filtration parameter eps.

        Parameters:
        eps: float - filtration parameter
        """
        self.edges = np.array([
            (i, j)
            for i, j in self.delauney_edges
            if np.linalg.norm(self.points[i] - self.points[j]) < 2 * eps
        ])

    def filter_two_simplices(self, eps):
        """
        Filter two simplices, with respect to the filtration parameter eps.

        Parameters:
        eps: float - filtration parameter
        """
        self.triangles = np.array([
            (c1, c2, c3)
            for c1, c2, c3 in self.delauney_triangles
            if all(
                np.linalg.norm(self.points[i] - self.points[j]) < 2 * eps
                for i, j in combinations([c1, c2, c3], 2)
            )
        ])