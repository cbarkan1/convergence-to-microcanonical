"""
Classes used for computing coarse-grained quantities.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

class Rho_numerical:
    """
    Numerical representation of rho^G, for performing numerical
    integrals and optimization (see Eq. 11).
    """
    def __init__(self,triangle,x_gridsize,y_gridsize):
        """
        Initialize meshgrids used for numerical integrals.

        Parameters:
            triangle (Triangle): Triangle object
            x_gridsize (int): number of gridpoints along x-axis
            y_gridsize (int): number of gridpoints along y-acis
        """
        self.L1l = triangle.L1l
        self.L1r = triangle.L1r
        self.L2 = triangle.L2
        self.Area = triangle.Area
        x_range = np.linspace(self.L1l,self.L1r,x_gridsize)
        y_range = np.linspace(0,self.L2,y_gridsize)
        self.x_mesh,self.y_mesh = np.meshgrid(x_range,y_range)
        x2_mesh,y2_mesh = self.x_mesh**2, self.y_mesh**2
        self.dA = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        self.array_of_mesh = np.array([self.x_mesh,self.y_mesh,x2_mesh,y2_mesh])

    def rho(self,lambdas):
        """
        Array of rho values, given lambdas of any length.

        Parameter:
            lambdas (np.array): Array containing lambda0,...,lambdaN
                                N=1,2,3,4 depending on which
                                bservables are used.

        Returns:
            np.array: Array of rho values.
        """
        N = len(lambdas) - 1
        lambdas_ax = lambdas[1:, np.newaxis, np.newaxis]
        exponent = -1 - lambdas[0] - np.sum(self.array_of_mesh[0:N,:,:] 
                                                        * lambdas_ax,axis=0)
        rho_array = np.exp(exponent) # Eq. 12

        #Mask values outside of triangle
        rho_array[self.y_mesh > self.L2*(1-self.x_mesh/self.L1l)] = 0.
        rho_array[self.y_mesh > self.L2*(1-self.x_mesh/self.L1r)] = 0.
        return rho_array

    def I(self,lambdas):
        """
        Calculate integral of rho over the triangle.

        Parameters:
            lambdas (np.array): Array of lambdas.

        Returns:
            float: Numerical intergral of rho over the triangle.
        """
        rho_array = self.rho(lambdas)
        return np.sum(rho_array) * self.dA

    def S_G(self,observed_vals,lambdas0):
        """
        Entropy of generalized coarse-grained distribution.

        Parameters:
            observed vals (list): Observed values. Length should be 
                        1,2,3, or 4, corresponding to which observables 
                        are observed.
            lambdas0 (np.array): initial lambdas array for 
                        optimization.
                        len(lambdas0) = 1 + len(observed_vals)

            Returns:
                tuple: (entropy value, optimization success, 
                        optimized lambdas)
        """
        obs = [1] + observed_vals

        def g(lambdas):
            """ g function (Eq. 13) """
            return self.I(lambdas)/self.Area + np.sum(obs*lambdas)

        result = minimize(g,lambdas0)
        lambdas = result.x
        return 1 + np.sum(obs*lambdas), result.success , np.array(lambdas)


class standardCG_9:
    """
    Used for computing standard coarse-grained quantities according
    to the partition shown in Fig. 1C.
    """
    def __init__(self,triangle):
        """
        Initializes the partition cells into the Areas array.
        """
        self.L1l = triangle.L1l
        self.L1r = triangle.L1r
        self.L2 = triangle.L2
        self.Afc = self.L1r/3 * self.L2/3  # Area of full cell
        self.Areas = self.Afc*np.ones((4,3))
        self.Areas[0,0] = self.Afc - (self.L2/3)*(-self.L1l/3)*0.5
        self.Areas[0,1] = ((-self.L1l*2/3) * (self.L2/3) 
                            - (self.L2/3)*(-self.L1l/3)*0.5)
        self.Areas[0,2] = (self.L2/3)*(-self.L1l/3)*0.5
        self.Areas[1,2] = self.Afc/2
        self.Areas[2,1] = self.Afc/2
        self.Areas[3,0] = self.Afc/2
        self.Total_Area = (0.5*(self.L1r-self.L1l)*self.L2)

    def rho_and_S(self,x_list,y_list):
        """
        compute the rho value in each cell (according to the standard
        coarse-graining), and the corresponding entropy.

        Parameters:
            x_list (list): List of particle x coordinates
            y_list (list): List of particle y coordinates

        Returns:
            np.array: Array of coarse-grained rho values
            float: entropy of coarse grained rho
        """
        num_particles = len(x_list)
        H,xedges,yedges = np.histogram2d(x_list,y_list,bins=[4,3],
                                        range=[[self.L1l,self.L1r],[0,self.L2]])
        rho_S = self.Total_Area*H/(self.Areas*num_particles)
        S_S = -1*np.sum(self.Areas*xlogy(rho_S,rho_S))
        return rho_S,S_S

#