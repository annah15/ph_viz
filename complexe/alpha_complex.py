from complexe import Complex
import numpy as np
import gudhi as gd
from itertools import combinations
from scipy.spatial import Delaunay


class Alpha_Complex(Complex):
    def __init__(self, points=None):
        super().__init__( points)
    
    def filter_simplices(self):
        """
        Filter simplices with respect to the alpha filtration.
        """
        alpha_skeleton = gd.AlphaComplex(points=self.points)
        self.simplextree = alpha_skeleton.create_simplex_tree()
        # Adapt filtration values match eps values of rips filtration
        for simplex, filtration in self.simplextree.get_filtration():
            self.simplextree.assign_filtration(simplex, 2*(filtration**0.5))
        self.simplextree.compute_persistence(persistence_dim_max=True)