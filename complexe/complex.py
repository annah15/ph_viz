import numpy as np
from itertools import combinations
import abc
import gudhi as gd

class Complex:
    def __init__(self, initial_epsilon, points=None):
        self.eps = initial_epsilon
        if points is None:
            self.points = np.random.rand(20, 2)*5
        else:
            self.points = points
        self.filter_one_simplices(initial_epsilon)
        self.filter_two_simplices(initial_epsilon)
        self.simplextree = gd.SimplexTree()
        for i in range(len(self.points)):
            self.simplextree.insert([i], 0)
        self.betti0, self.betti1 = self.compute_betti(initial_epsilon)
    
    @abc.abstractmethod
    def filter_one_simplices(self, eps:float):
        """
        Filter one simplices, with respect to the filtration parameter eps.

        Parameters:
        eps: float - filtration parameter
        """
        return

    @abc.abstractmethod
    def filter_two_simplices(self, eps):
        """
        Filter two simplices, with respect to the filtration parameter eps.

        Parameters:
        eps: float - filtration parameter
        """
        return 
    
    def compute_betti(self, eps):
        for edge in self.edges:
            self.simplextree.insert(list(edge))
        for triangle in self.triangles:
            self.simplextree.insert(list(triangle))
        # Compute the persistence diagram
        self.simplextree.compute_persistence()

        # Extract persistence intervals for dimensions 0 and 1
        betti_0_intervals = self.simplextree.persistence_intervals_in_dimension(0)
        betti_1_intervals = self.simplextree.persistence_intervals_in_dimension(1)

        # Calculate Betti numbers
        betti_0 = len([interval for interval in betti_0_intervals if interval[1] == float('inf')])
        betti_1 = len([interval for interval in betti_1_intervals if interval[1] == float('inf')])
        return betti_0, betti_1
