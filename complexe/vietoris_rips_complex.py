import numpy as np
import gudhi as gd
from scipy.spatial import distance_matrix
from itertools import combinations
from complexe import Complex

class VR_Complex(Complex):
    def __init__(self, points=None):
        super().__init__(points)

    def filter_simplices(self):
        """
        Filter simplices with respect to the vietoris-rips filtration up to dimension 2.
        """
        rips_skeleton = gd.RipsComplex(points=self.points)
        self.simplextree = rips_skeleton.create_simplex_tree(max_dimension=2)
        self.simplextree.compute_persistence(persistence_dim_max=True)