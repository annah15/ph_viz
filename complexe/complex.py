import numpy as np
from itertools import combinations
import abc
import gudhi as gd
from pprint import pprint

class Complex:
    def __init__(self, points=None):
        if points is None:
            self.points = np.random.rand(20, 2)*5
        else:
            self.points = points
        self.simplextree = gd.SimplexTree()
        self.simplextree.set_dimension(2)
        for i in range(len(self.points)):
            self.simplextree.insert([i], 0)
        self.betti = [len(self.points), 0, 0]
    
    @abc.abstractmethod
    def filter_simplices(self):
        """
        Filter simplices with respect to a filtration up to dimension 2.
        """
        return
    
    def get_edges(self, eps=None):
        '''
        Extract edges from the simplicial complex
        
        Args:
            - eps (float), optional: Maximum distance for the edges
            
        Returns:
            - edges (np.array): Edges of the simplicial complex (with optional distance constraint)
        '''
        if eps is None:
            return np.array([simplex[0] for simplex in self.simplextree.get_skeleton(1) if len(simplex[0]) == 2])
        else:
            return np.array([simplex[0] for simplex in self.simplextree.get_skeleton(1) if len(simplex[0]) == 2 and simplex[1] <= 2*eps])
    
    def get_triangles(self, eps=None):
        '''
        Extract triangles from the simplicial complex

        Args:
            - eps (float), optional: Maximum filtration for the triangles

        Returns:
            - triangles (np.array): Triangles of the simplicial complex (with optional filtration constraint)
        '''
        if eps is None:
            return np.array([simplex[0] for simplex in self.simplextree.get_skeleton(2) if len(simplex[0]) == 3])
        else:
            return np.array([simplex[0] for simplex in self.simplextree.get_skeleton(2) if len(simplex[0]) == 3 and simplex[1] <= 2*eps])
        
    def compute_betti(self, eps):
        '''
        Compute the Betti numbers of the simplicial complex

        Args:
            - eps (float): Filtration value for the Betti numbers

        Returns:
            - betti (np.array): Betti numbers of the simplicial complex
        '''
        return np.pad(self.simplextree.persistent_betti_numbers(2*eps, 2*eps), (0, 3), 'constant', constant_values=(0))

    def get_persistence_pairs(self, eps=None, extended=False):
        '''
        Extract persistence pairs (birth and death values)

        Args:
            - eps (float), optional: Maximum distance for the persistence pairs
            - extended (bool), optional: Compute extended persistence

        Returns:
            - dimensions (dict):  Persistence pairs for each dimension
        '''
        max_filtration = max([death for _, (_, death) in self.simplextree.persistence() if death != float('inf')]+[2])+1
        # Prepare data structures
        diagrams = []
        # List of ordinary, extended and relative persistence diagrams
        for _ in range(3):
            # With dictionaries for each dimension
            dimensions = {}
            for dim in range(4):
                    dimensions[dim] = dict(
                                    x = [],
                                    y = []
                    )
            diagrams.append(dimensions)

        if not extended:
            # Extract persistence pairs (birth and death values)
            persistence = self.simplextree.persistence()

            for interval in persistence:
                dim, (birth, death) = interval
                if eps==None or (birth < 2*eps and death < 2*eps):
                    diagrams[0][dim]['x'].append(birth)
                    diagrams[0][dim]['y'].append(death if death != float('inf') else max_filtration)
        
        else:
            # Extend the simplicial complex by adding a vertex and connecting it to all simplices of the original complex (coning)
            cone = gd.SimplexTree(self.simplextree)
            cone.insert([len(self.points)], filtration=max_filtration)
            for simplex, filt in self.simplextree.get_simplices():
                # To extract the diffent types of persistence diagrams, we need to shift the filtration values
                cone.assign_filtration(simplex, filtration=filt - max_filtration)
                cone.insert(simplex + [len(self.points)], filtration=filt)
            
            # Compute persistence pairs of the extended complex
            persistence = cone.persistence()
            for interval in persistence:
                dim, (birth, death) = interval
                if eps==None or (birth < 2*eps and death < 2*eps):
                    if (birth < 0 and death < 0):
                        #The interval is in the ordinary persistence diagram
                        diagrams[0][dim]['x'].append(birth + max_filtration)
                        diagrams[0][dim]['y'].append(death + max_filtration )
                    if (birth < 0 and death >= 0):
                        #The interval is in the essential persistence diagram
                        diagrams[1][dim]['x'].append(birth + max_filtration)
                        diagrams[1][dim]['x'].append(death if death != float('inf') else max_filtration)
                        diagrams[1][dim]['y'].append(death if death != float('inf') else max_filtration)
                        diagrams[1][dim]['y'].append(birth + max_filtration)
                    if (birth >= 0 and birth < max_filtration and death >= 0 and death < max_filtration):
                        #The interval is in the relative persistence diagram
                        diagrams[2][dim-1]['y'].append(birth if birth != float('inf') else max_filtration)
                        diagrams[2][dim-1]['x'].append(death if death != float('inf') else max_filtration)    
        return diagrams
            

    
            