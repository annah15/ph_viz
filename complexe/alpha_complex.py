from complexe import Complex
import numpy as np
import gudhi as gd
from scipy.spatial import Voronoi


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
    
    def compute_voronoi(self):
        '''
        Compute the Voronoi diagram of the points
        
        Returns:
            - segments_xs (list): x-coordinates of the Voronoi diagram segments
            - segments_ys (list): y-coordinates of the Voronoi diagram segments
        '''
        #if the list of points is empty, there is no Voronoi diagram to compute, return empty lists
        if len(self.points) <= 2:
            return [], []
        
        vor = Voronoi(self.points)
        # Prepare data structures
        segments_xs, segments_ys = [], []

        # Compute bounds
        center = vor.points.mean(axis=0)
        ptp_bound = np.ptp(vor.points, axis=0)

        # Extract finite and infinite segments
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):  # Finite segment
                segments_xs.append(vor.vertices[simplex, 0])
                segments_ys.append(vor.vertices[simplex, 1])
            else:  # Infinite segment
                i = simplex[simplex >= 0][0]
                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal
                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[i] + direction * ptp_bound.max() * 2
                segments_xs.append(np.array([vor.vertices[i], far_point])[:, 0])
                segments_ys.append(np.array([vor.vertices[i], far_point])[:, 1])      
        return segments_xs, segments_ys