"""
 ==============================================================================
 Copyright (C) 2016--2017 Michal Kalkowski (MIT License)
 kalkowski.m@gmail.com

 This is a part of the axisafe package developed for simulating elastic
 wave propagation in buried/submerged fluid-filled waveguides. The package is 
 based on the publication:
     Kalkowski MK et al. Axisymmetric semi-analytical finite elements for modelling 
     waves in buried/submerged fluid-filled waveguides. Comput Struct (2017), 
     https://doi.org/10.1016/j.compstruc.2017.10.004

This file contains element definitions.
 ==============================================================================
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from . import shape_functions as shape_fun

def gamma_PML(x, gamma, PML_start, PML_thk):
    """
    Polynomial stretching profile for a perfectly matched layer.

    Parameters:
        x : physical coordinate
        gamma : average value of the profile
        PML_start : where the PML starts
        PML_thk : thickness of the PML

    Returns:
        the value of the profile function at x
    """
    return 1 + 3*(gamma - 1)*((abs(x - PML_start))/PML_thk)**2

def gamma_PML_exp(x, gamma, PML_start, PML_thk):
    """
    Polynomial stretching profile for a perfectly matched layer.

    Parameters:
        x : physical coordinate
        gamma : profile parameters in the form a + 1j*b
        PML_start : where the PML starts
        PML_thk : thickness of the PML

    Returns:
        the value of the profile function at x
    """
    local_x = (x - PML_start)/PML_thk
    return np.exp(gamma.real*local_x) - 1j*(np.exp(gamma.imag*local_x)- 1)


class SLAX6(object):
    """
    Class for an axisymmetric 1D line element with nodes wrt GLL quadrature
    and higher order Lagrange interpolating polynomials
    u = [ur utheta uz]
    strain directions = [rr tt zz tz rz rt]
    n is the circumferential order number
    Here six matrices are assembled and the circumferential order is applied
    at a later stage
    """
    def __init__(self, nodes_at, n):
        """
        Initialise the element with node locations.

        Parameters:
            nodes_at : node locations
        """

        self.n = n
        self.nodes_at = nodes_at
        self.nodes_per_el = len(nodes_at)
        self.no_of_dofs = sum([3]*self.nodes_per_el)
        # define basic properties of the element (no of nodes, etc.)
        self.dofs_per_node = [3]*self.nodes_per_el
        self.dofs_domain = [['s']*3]*self.nodes_per_el
        self.dofs_per_el = sum(self.dofs_per_node)
        # define T transormation component
        self.T_components = [[1, 1j, 1j]]*self.nodes_per_el

        self.C = None
        self.loss_factor = None
        self.rho = None
        # Pre-allocate the arrays
        self.K_Fz = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_Ft = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_F = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_F1 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_F2 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_F3 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_1 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_2 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_3 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_4 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_5 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_6 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.M = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex64')
        self.Hs0 = None
        self.Hs1 = None

    def add_properties(self, list_of_props):
        """Assign mechanical properties and the circumferential order.

        Parameters:
            list_of_props : [lame_1, lame_2, denisty, loss_factor]
            n : circumferential order
        """
        # assign the properties
        self.loss_factor = list_of_props[3]
        lame_1 = list_of_props[0]
        lame_2 = list_of_props[1]
        self.C = (1 + self.loss_factor*1.0j)*\
                np.array([(lame_1 + 2*lame_2, lame_1, lame_1, 0, 0, 0),
                          (lame_1, lame_1 + 2*lame_2, lame_1, 0, 0, 0),
                          (lame_1, lame_1, lame_1 + 2*lame_2, 0, 0, 0),
                          (0, 0, 0, lame_2, 0, 0),
                          (0, 0, 0, 0, lame_2, 0),
                          (0, 0, 0, 0, 0, lame_2)])
        self.rho = list_of_props[2]

    def inspect_shape_fun(self):
        """
        This function computes and plots shape functions in the natural coordinates
        together with their derivatives.

        (not updated for long)
        """
        ksi, _ = shape_fun.build_GLL_quadrature(len(self.nodes_at))
        xi_dense = np.linspace(-1, 1, 100)
        N, dN = shape_fun.lagrange_poly(xi_dense, len(ksi))
        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1_limits = 1.2*np.min(N), 1.2*np.max(N)
        ax2_limits = 1.2*np.min(dN), 1.2*np.max(dN)
        ax2.set_xlabel(r'$\xi$')
        ax1.set_xlabel(r'shape functions')
        ax2.set_xlabel(r'shape functions derivatives')
        for this_xi in ksi:
            ax1.plot(2*[this_xi], list(ax1_limits), '--', c='red')
            ax2.plot(2*[this_xi], list(ax2_limits), '--', c='red')
        ax1.plot(xi_dense, N.T)
        ax2.plot(xi_dense, dN.T)
        ax1.set_title(r'Spectral element of order ' + \
                      str(len(self.nodes_at) - 1) + ' with ' +\
                      str(len(self.nodes_at)) + ' nodes.')

    def calculate_matrices(self):
        """Calculates the shape function and B matrices including
        Jacobian*Gauss Weight* det(Jacobian) at all Gauss quadrature points"""

        # get the nodal locations weights and shape functions for a given GLL quadrature
        ksis, weights, NN, NN_r, JJ = \
                            shape_fun.line_spectral_GLL_lagrange(self.nodes_at)

        # extract the matrices and sum at each integration point
        for i in range(len(ksis)):

            # shape funcion values at current location are NN[:, i]
            # generate the shape function matrix in standard form
            diagonals = [[funct]*3 for funct in NN[:, i]]
            N = np.column_stack([np.diag(diagonal) for diagonal in diagonals])

            diagonals_r = [[funct]*3 for funct in NN_r[:, i]]
            N_r = np.column_stack([np.diag(diagonal) for diagonal in diagonals_r])

            # local radius in physical coordinates
            r_i = self.nodes_at.dot(NN[:, i])

            # differential operator matrices
            L_z = np.zeros([6, 3])
            L_theta = np.zeros([6, 3])
            L_r = np.zeros([6, 3])
            L = np.zeros([6, 3])
            L_z[2, 2], L_z[3, 1], L_z[4, 0] = 1, 1, 1
            L_r[0, 0], L_r[4, 2], L_r[5, 1] = 1, 1, 1
            L[1, 0], L[5, 1] = 1.0, -1.0
            L_theta[1, 1], L_theta[3, 2], L_theta[5, 0] = 1, 1, 1


            B_1 = L.dot(N)/r_i + L_r.dot(N_r)
            B_2 = L_theta.dot(N)/r_i
            B_3 = L_z.dot(N)

            K_F1 = B_1.T.dot(self.C).dot(B_3)
            K_F2 = B_1.T.dot(self.C).dot(B_2)
            K_F3 = B_2.T.dot(self.C).dot(B_3)

            K_1 = B_1.T.dot(self.C).dot(B_1)
            K_2 = K_F2.T - K_F2
            K_3 = K_F1.T - K_F1
            K_4 = K_F3.T + K_F3
            K_5 = B_2.T.dot(self.C).dot(B_2)
            K_6 = B_3.T.dot(self.C).dot(B_3)

            K_F = K_F1.T
            K_Ft = K_F3.T
            K_Fz = K_6

            M = self.rho*N.T.dot(N)

            # perform the integration
            self.K_F += 2*np.pi*weights[i]*K_F*JJ[i]*r_i
            self.K_Ft += 2*np.pi*weights[i]*K_Ft*JJ[i]*r_i
            self.K_Fz += 2*np.pi*weights[i]*K_Fz*JJ[i]*r_i

            self.K_F1 += 2*np.pi*weights[i]*K_F1*JJ[i]*r_i
            self.K_F2 += 2*np.pi*weights[i]*K_F2*JJ[i]*r_i
            self.K_F3 += 2*np.pi*weights[i]*K_F3*JJ[i]*r_i

            self.K_1 += 2*np.pi*weights[i]*K_1*JJ[i]*r_i
            self.K_2 += 2*np.pi*weights[i]*K_2*JJ[i]*r_i
            self.K_3 += 2*np.pi*weights[i]*K_3*JJ[i]*r_i
            self.K_4 += 2*np.pi*weights[i]*K_4*JJ[i]*r_i
            self.K_5 += 2*np.pi*weights[i]*K_5*JJ[i]*r_i
            self.K_6 += 2*np.pi*weights[i]*K_6*JJ[i]*r_i
            self.M += 2*np.pi*weights[i]*M*JJ[i]*r_i

        # if the element is coupled to fluid elements - create the ingredients
        # of the coupling matrices
        normal = np.array([1, 0, 0]).reshape(-1, 1)

        diagonals = [[funct]*3 for funct in NN[:, 0]]
        N_u = np.column_stack([np.diag(diagonal) for diagonal in diagonals])
        self.Hs0 = 2*np.pi*N_u.T.dot(normal)

        diagonals = [[funct]*3 for funct in NN[:, -1]]
        N_u = np.column_stack([np.diag(diagonal) for diagonal in diagonals])
        self.Hs1 = 2*np.pi*N_u.T.dot(normal)*self.nodes_at[-1]

class ALAX6(object):
    """
    Class for an acoustic axisymmetric 1D line element with nodes wrt GLL quadrature
    and higher order Lagrange interpolating polynomials
    Each node has one degree of freedom- velocity potential
    n is the circumferential order number"""
    def __init__(self, nodes_at, n):
        """
        Initialise the element with node locations.

        Parameters:
            nodes_at : node locations
        """

        self.nodes_at = nodes_at
        self.nodes_per_el = len(nodes_at)
        self.no_of_dofs = self.nodes_per_el
        # define basic properties of the element
        self.dofs_per_node = [1]*self.nodes_per_el
        self.dofs_domain = [['f']]*self.nodes_per_el
        self.dofs_per_el = sum(self.dofs_per_node)
        # define T-transformation components
        self.T_components = [[1]]*self.nodes_per_el

        self.c_p = None
        self.rho = None
        self.n = n
        # Pre-allocate the arrays
        self.K_Fz = np.zeros([self.no_of_dofs, self.no_of_dofs],
                             dtype='complex')
        self.K_Ft = np.zeros([self.no_of_dofs, self.no_of_dofs],
                             dtype='complex')
        self.K_F = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex')
        self.K_1 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dz*dz
        self.K_2 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dtheta*dz
        self.K_3 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dz
        self.K_4 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dtheta*dtheta
        self.K_5 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dtheta
        self.K_6 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # -
        self.M = np.zeros([self.no_of_dofs, self.no_of_dofs],
                          dtype='float64')
        self.Hf0 = None
        self.Hf1 = None

    def add_properties(self, list_of_props):
        """Assign mechanical properties.

        Parameters:
            list_of_props : [bulk modulus, density]

        """
        # assign properties
        self.c_p = (list_of_props[0]/list_of_props[1])**0.5
        self.rho = list_of_props[1]

    def calculate_matrices(self):
        """Calculates the shape function and b matrices including
        Jacobian*Gauss Weight* det(Jacobian) at all Gauss quadrature points"""
        # get the nodal locations weights and shape functions for a given GLL quadrature

        ksis, weights, NN, NN_r, JJ = \
                            shape_fun.line_spectral_GLL_lagrange(self.nodes_at)

        # extract the matrices and sum at each integration point
        for i in range(len(ksis)):

            # shape funcion values at current location are NN[:, i]
            # generate the shape function matrix in standard form
            N = NN[:, i].reshape(1, -1)
            N_r = NN_r[:, i].reshape(1, -1)
            # local radius in physical coordinates
            r_i = self.nodes_at.dot(NN[:, i])

            K_1 = N_r.T.dot(N_r)
            K_5 = 1/r_i**2*N.T.dot(N)
            K_6 = N.T.dot(N)
            M = N.T.dot(N)/self.c_p**2

            # perform the integration
            self.K_6 += -self.rho*2*np.pi*weights[i]*K_6*JJ[i]*r_i
            self.K_1 += -self.rho*2*np.pi*weights[i]*K_1*JJ[i]*r_i
            self.K_5 += -self.rho*2*np.pi*weights[i]*K_5*JJ[i]*r_i
            self.K_Fz += -self.rho*2*np.pi*weights[i]*K_6*JJ[i]*r_i
            self.K_Ft += -self.rho*2*np.pi*weights[i]*K_5*JJ[i]*r_i
            self.M += -self.rho*2*np.pi*weights[i]*M*JJ[i]*r_i

        # if the element is coupled to solid elements - create the ingredients
        # of the coupling matrices
        N_phi = NN[:, 0].reshape(-1, 1)
        self.Hf0 = self.rho*N_phi

        N_phi = NN[:, -1].reshape(-1, 1)
        self.Hf1 = self.rho*N_phi*self.nodes_at[-1]

class SLAX6_core(object):
    """
    Class for an axisymmetric core (with a node at the axis of symmetry)
    1D line element with nodes wrt GLJ quadrature and higher order Lagrange
    interpolating polynomials
    u = [ur utheta uz]
    strain directions = [rr tt zz tz rz rt]
    n is the circumferential order number
    Here six matrices are assembled and the circumferential order is applied
    at a later stage"""
    def __init__(self, nodes_at, n):
        """
        Initialise the element with node locations.

        Parameters:
            nodes_at : node locations
        """
        self.nodes_at = nodes_at
        self.nodes_per_el = len(nodes_at)
        # double-check if there is a node at the axis of symmetry
        # specify basic element characteristics given the circumferential order
        if 0 in self.nodes_at:
            if n == 0:
                self.dofs_per_node = [1] + [3]*(self.nodes_per_el - 1)
                self.dofs_domain = [['s']*1] + [['s']*3]*(self.nodes_per_el - 1)
                self.dofs_per_el = sum(self.dofs_per_node)
                self.T_components = [[1j]] + [[1, 1j, 1j]]*(self.nodes_per_el - 1)
            elif n == 1:
                self.dofs_per_node = [2] + [3]*(self.nodes_per_el - 1)
                self.dofs_domain = [['s']*2] + [['s']*3]*(self.nodes_per_el - 1)
                self.dofs_per_el = sum(self.dofs_per_node)
                self.T_components = [[1, 1j]] + [[1, 1j, 1j]]*(self.nodes_per_el - 1)
            elif n > 1:
                self.dofs_per_node = [0] + [3]*(self.nodes_per_el - 1)
                self.dofs_domain = [[]] + [['s']*3]*(self.nodes_per_el - 1)
                self.dofs_per_el = sum(self.dofs_per_node)
                self.T_components = [[]] + [[1, 1j, 1j]]*(self.nodes_per_el - 1)
        else:
            self.dofs_per_node = [3]*self.nodes_per_el
            self.dofs_domain = [['s']*3]*self.nodes_per_el
            self.dofs_per_el = sum(self.dofs_per_node)
            self.T_components = [[1, 1j, 1j]]*self.nodes_per_el
        self.n = n

        self.C = None
        self.loss_factor = None
        self.rho = None

        # preallocate the arrays
        # use full size - respective dofs will be rmoved at a later stage
        no_of_dofs = sum([3]*self.nodes_per_el)
        # keep the attribute no of dofs not set
        self.no_of_dofs = None
        self.K_Fz = np.zeros([no_of_dofs, no_of_dofs], dtype='complex')
        self.K_Ft = np.zeros([no_of_dofs, no_of_dofs], dtype='complex')
        self.K_F = np.zeros([no_of_dofs, no_of_dofs], dtype='complex')
        self.K_F1 = np.zeros([no_of_dofs, no_of_dofs], dtype='complex')
        self.K_F2 = np.zeros([no_of_dofs, no_of_dofs], dtype='complex')
        self.K_F3 = np.zeros([no_of_dofs, no_of_dofs], dtype='complex')
        self.K_1 = np.zeros([no_of_dofs, no_of_dofs], dtype='complex')
        self.K_2 = np.zeros([no_of_dofs, no_of_dofs], dtype='complex')
        self.K_3 = np.zeros([no_of_dofs, no_of_dofs], dtype='complex')
        self.K_4 = np.zeros([no_of_dofs, no_of_dofs], dtype='complex')
        self.K_5 = np.zeros([no_of_dofs, no_of_dofs], dtype='complex')
        self.K_6 = np.zeros([no_of_dofs, no_of_dofs], dtype='complex')
        self.M = np.zeros([no_of_dofs, no_of_dofs], dtype='complex64')
        self. Hs1 = None

    def add_properties(self, list_of_props):
        """Assign mechanical properties and the circumferential order.

        Parameters:
            list_of_props : [lame_1, lame_2, denisty, loss_factor]
            n : circumferential order
        """
        # assign the properties
        self.loss_factor = list_of_props[3]
        lame_1 = list_of_props[0]
        lame_2 = list_of_props[1]
        self.C = (1 + self.loss_factor*1.0j)*\
                np.array([(lame_1 + 2*lame_2, lame_1, lame_1, 0, 0, 0),
                          (lame_1, lame_1 + 2*lame_2, lame_1, 0, 0, 0),
                          (lame_1, lame_1, lame_1 + 2*lame_2, 0, 0, 0),
                          (0, 0, 0, lame_2, 0, 0),
                          (0, 0, 0, 0, lame_2, 0),
                          (0, 0, 0, 0, 0, lame_2)])
        self.rho = list_of_props[2]

    def inspect_shape_fun(self):
        """
        This function computes and plots shape functions in the natural coordinates
        together with their derivatives.

        (not updated for long)
        """
        ksi, _ = shape_fun.build_GLJ_quadrature(len(self.nodes_at))
        xi_dense = np.linspace(-1, 1, 100)
        N, _ = shape_fun.lagrange_GLJ(xi_dense, len(ksi))
        dN = np.column_stack([splev(xi_dense, splrep(xi_dense, N[:, i]), der=1) \
                          for i in range(N.shape[1])])
        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1_limits = 1.2*np.min(N), 1.2*np.max(N)
        ax2_limits = 1.2*np.min(dN), 1.2*np.max(dN)
        ax2.set_xlabel(r'$\xi$')
        ax1.set_xlabel(r'shape functions')
        ax2.set_xlabel(r'shape functions derivatives')
        for this_xi in ksi:
            ax1.plot(2*[this_xi], list(ax1_limits), '--', c='red')
            ax2.plot(2*[this_xi], list(ax2_limits), '--', c='red')
        ax1.plot(xi_dense, N)
        ax2.plot(xi_dense, dN)
        ax1.set_title(r'Spectral element of order ' + \
                      str(len(self.nodes_at) - 1) + ' with ' +\
                      str(len(self.nodes_at)) + ' nodes.')

    def calculate_matrices(self):
        """Calculates the shape function and B matrices including
        Jacobian*Gauss Weight* det(Jacobian) at all Gauss quadrature points"""


        # get the nodal locations weights and shape functions for a given GLJ quadrature
        ksis, weights, NN, NN_r, JJ = \
                            shape_fun.line_spectral_GLJ_lagrange(self.nodes_at)

        # extract the matrices and sum at each integration point
        for i, ksi in enumerate(list(ksis)):

            # shape funcion values at current location are NN[:, i]
            # generate the shape function matrix in standard form
            diagonals = [[funct]*3 for funct in NN[:, i]]
            N = np.column_stack([np.diag(diagonal) for diagonal in diagonals])

            diagonals_r = [[funct]*3 for funct in NN_r[:, i]]
            N_r = np.column_stack([np.diag(diagonal) for diagonal in diagonals_r])

            # local r in physical coordinates
            r_i = self.nodes_at[i]
            # scale the radius according to the de'l'Hospital's rule for the axis
            # singularity
            w = 1 + ksi
            if np.round(r_i, 7) == 0:
                rat = JJ[i]
            else:
                rat = r_i/w

            # differential operator matrices
            L_z = np.zeros([6, 3])
            L_theta = np.zeros([6, 3])
            L_r = np.zeros([6, 3])
            L = np.zeros([6, 3])
            L_z[2, 2], L_z[3, 1], L_z[4, 0] = 1, 1, 1
            L_r[0, 0], L_r[4, 2], L_r[5, 1] = 1, 1, 1
            L[1, 0], L[5, 1] = 1.0, -1.0
            L_theta[1, 1], L_theta[3, 2], L_theta[5, 0] = 1, 1, 1

            # use different forms for the axis of symmetry and the rest of the domain
            if np.round(r_i, 7) == 0:
                B_1 = L.dot(N_r) + L_r.dot(N_r)
                B_2 = L_theta.dot(N_r)
                B_3 = L_z.dot(N)
            else:
                B_1 = L.dot(N)/r_i + L_r.dot(N_r)
                B_2 = L_theta.dot(N)/r_i
                B_3 = L_z.dot(N)

            K_F1 = B_1.T.dot(self.C).dot(B_3)
            K_F2 = B_1.T.dot(self.C).dot(B_2)
            K_F3 = B_2.T.dot(self.C).dot(B_3)

            K_1 = B_1.T.dot(self.C).dot(B_1)
            K_2 = K_F2.T - K_F2
            K_3 = K_F1.T - K_F1
            K_4 = K_F3.T + K_F3
            K_5 = B_2.T.dot(self.C).dot(B_2)
            K_6 = B_3.T.dot(self.C).dot(B_3)

            K_F = K_F1.T
            K_Ft = K_F3.T
            K_Fz = K_6
            M = self.rho*N.T.dot(N)

            # GLJ integration
            self.K_F += 2*np.pi*weights[i]*K_F*JJ[i]*rat
            self.K_Ft += 2*np.pi*weights[i]*K_Ft*JJ[i]*rat
            self.K_Fz += 2*np.pi*weights[i]*K_Fz*JJ[i]*rat

            self.K_F1 += 2*np.pi*weights[i]*K_F1*JJ[i]*rat
            self.K_F2 += 2*np.pi*weights[i]*K_F2*JJ[i]*rat
            self.K_F3 += 2*np.pi*weights[i]*K_F3*JJ[i]*rat

            self.K_1 += 2*np.pi*weights[i]*K_1*JJ[i]*rat
            self.K_2 += 2*np.pi*weights[i]*K_2*JJ[i]*rat
            self.K_3 += 2*np.pi*weights[i]*K_3*JJ[i]*rat
            self.K_4 += 2*np.pi*weights[i]*K_4*JJ[i]*rat
            self.K_5 += 2*np.pi*weights[i]*K_5*JJ[i]*rat
            self.K_6 += 2*np.pi*weights[i]*K_6*JJ[i]*rat

            self.M += 2*np.pi*weights[i]*M*JJ[i]*rat

        # if the element is coupled to fluid elements - create the ingredients
        # of the coupling matrices
        normal = np.array([1, 0, 0]).reshape(-1, 1)
        diagonals = [[funct]*3 for funct in NN[:, -1]]
        N_u = np.column_stack([np.diag(diagonal) for diagonal in diagonals])
        self.Hs1 = 2*np.pi*N_u.T.dot(normal)*self.nodes_at[-1]

        # apply the boundary conditions at the axis of symmetry
        if 0 in self.nodes_at:
            if self.n == 0:
                self.K_1 = np.delete(np.delete(self.K_1, [0, 1], 0), [0, 1], 1)
                self.K_2 = np.delete(np.delete(self.K_2, [0, 1], 0), [0, 1], 1)
                self.K_3 = np.delete(np.delete(self.K_3, [0, 1], 0), [0, 1], 1)
                self.K_4 = np.delete(np.delete(self.K_4, [0, 1], 0), [0, 1], 1)
                self.K_5 = np.delete(np.delete(self.K_5, [0, 1], 0), [0, 1], 1)
                self.K_6 = np.delete(np.delete(self.K_6, [0, 1], 0), [0, 1], 1)
                self.M = np.delete(np.delete(self.M, [0, 1], 0), [0, 1], 1)
                self.K_F = np.delete(np.delete(self.K_F, [0, 1], 0), [0, 1], 1)
                self.K_Ft = np.delete(np.delete(self.K_Ft, [0, 1], 0), [0, 1], 1)
                self.K_Fz = np.delete(np.delete(self.K_Fz, [0, 1], 0), [0, 1], 1)
                self.Hs1 = 2*np.pi*N_u[:, 2:].T.dot(normal)*self.nodes_at[-1]
            elif self.n == 1:
                self.K_1 = np.delete(np.delete(self.K_1, [2], 0), [2], 1)
                self.K_2 = np.delete(np.delete(self.K_2, [2], 0), [2], 1)
                self.K_3 = np.delete(np.delete(self.K_3, [2], 0), [2], 1)
                self.K_4 = np.delete(np.delete(self.K_4, [2], 0), [2], 1)
                self.K_5 = np.delete(np.delete(self.K_5, [2], 0), [2], 1)
                self.K_6 = np.delete(np.delete(self.K_6, [2], 0), [2], 1)
                self.M = np.delete(np.delete(self.M, [2], 0), [2], 1)
                self.K_F = np.delete(np.delete(self.K_F, [2], 0), [2], 1)
                self.K_Ft = np.delete(np.delete(self.K_Ft, [2], 0), [2], 1)
                self.K_Fz = np.delete(np.delete(self.K_Fz, [2], 0), [2], 1)
                self.Hs1 = 2*np.pi*np.delete(N_u, [2], 1).T.dot(normal)*self.nodes_at[-1]
            elif self.n > 1:
                self.K_1 = np.delete(np.delete(self.K_1, [0, 1, 2], 0),
                                     [0, 1, 2], 1)
                self.K_2 = np.delete(np.delete(self.K_2, [0, 1, 2], 0),
                                     [0, 1, 2], 1)
                self.K_3 = np.delete(np.delete(self.K_3, [0, 1, 2], 0),
                                     [0, 1, 2], 1)
                self.K_4 = np.delete(np.delete(self.K_4, [0, 1, 2], 0),
                                     [0, 1, 2], 1)
                self.K_5 = np.delete(np.delete(self.K_5, [0, 1, 2], 0),
                                     [0, 1, 2], 1)
                self.K_6 = np.delete(np.delete(self.K_6, [0, 1, 2], 0),
                                     [0, 1, 2], 1)
                self.M = np.delete(np.delete(self.M, [0, 1, 2], 0),
                                   [0, 1, 2], 1)
                self.K_F = np.delete(np.delete(self.K_F, [0, 1, 2], 0),
                                     [0, 1, 2], 1)
                self.K_Ft = np.delete(np.delete(self.K_Ft, [0, 1, 2], 0),
                                      [0, 1, 2], 1)
                self.K_Fz = np.delete(np.delete(self.K_Fz, [0, 1, 2], 0),
                                      [0, 1, 2], 1)
                self.Hs1 = 2*np.pi*N_u[:, 3:].T.dot(normal)*self.nodes_at[-1]
            self.no_of_dofs = self.K_1.shape[0]

class ALAX6_core(object):
    """
    Class for an acoustic axisymmetric 1D line core element (with a node at the axis
    of symmetry) with nodes wrt GLJ quadrature and higher order Lagrange
    interpolating polynomials. Each node has one degree of freedom- velocity potential
    n is the circumferential order number"""

    def __init__(self, nodes_at, n):
        """
        Initialise the element with node locations.

        Parameters:
            nodes_at : node locations
        """

        self.nodes_at = nodes_at
        self.nodes_per_el = len(nodes_at)
        self.no_of_dofs = self.nodes_per_el
        self.n = n
        # doulbe-check if there is a node at the axis of symmetry
        # define basic properties of the element and T-transformation
        if 0 in self.nodes_at and n != 0:
            self.dofs_per_node = [0] + [1]*(self.nodes_per_el - 1)
            self.dofs_domain = [[]] + [['f']]*(self.nodes_per_el - 1)
            self.dofs_per_el = sum(self.dofs_per_node)
            self.T_components = [[]] + [[1]]*(self.nodes_per_el - 1)
        else:
            self.dofs_per_node = [1]*self.nodes_per_el
            self.dofs_domain = [['f']]*self.nodes_per_el
            self.dofs_per_el = sum(self.dofs_per_node)
            self.T_components = [[1]]*self.nodes_per_el

        self.c_p = None
        self.rho = None
        # preallocate the arrays
        self.K_Fz = np.zeros([self.no_of_dofs, self.no_of_dofs],
                             dtype='complex')
        self.K_Ft = np.zeros([self.no_of_dofs, self.no_of_dofs],
                             dtype='complex')
        self.K_F = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex')
        self.K_1 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dz*dz
        self.K_2 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dtheta*dz
        self.K_3 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dz
        self.K_4 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dtheta*dtheta
        self.K_5 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dtheta
        self.K_6 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # -
        self.M = np.zeros([self.no_of_dofs, self.no_of_dofs],
                          dtype='float64')
        self.Hf1 = None

    def add_properties(self, list_of_props):
        """Assign mechanical properties.

        Parameters:
            list_of_props : [bulk modulus, density]

        """
        # assign properties
        self.c_p = (list_of_props[0]/list_of_props[1])**0.5
        self.rho = list_of_props[1]

    def inspect_shape_fun(self):
        """
        This function computes and plots shape functions in the natural coordinates
        together with their derivatives

        (not updated for long)
        """
        ksi, _ = shape_fun.build_GLJ_quadrature(len(self.nodes_at))
        xi_dense = np.linspace(-1, 1, 100)
        N, _ = shape_fun.lagrange_GLJ(xi_dense, len(ksi))
        dN = np.column_stack([splev(xi_dense, splrep(xi_dense, N[:, i]), der=1) \
                          for i in range(N.shape[1])])
        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1_limits = 1.2*np.min(N), 1.2*np.max(N)
        ax2_limits = 1.2*np.min(dN), 1.2*np.max(dN)
        ax2.set_xlabel(r'$\xi$')
        ax1.set_xlabel(r'shape functions')
        ax2.set_xlabel(r'shape functions derivatives')
        for this_xi in ksi:
            ax1.plot(2*[this_xi], list(ax1_limits), '--', c='red')
            ax2.plot(2*[this_xi], list(ax2_limits), '--', c='red')
        ax1.plot(xi_dense, N)
        ax2.plot(xi_dense, dN)
        ax1.set_title(r'Spectral element of order ' + \
                      str(len(self.nodes_at) - 1) + ' with ' +\
                      str(len(self.nodes_at)) + ' nodes.')

    def calculate_matrices(self):
        """Calculates the shape function and b matrices including
        Jacobian*Gauss Weight* det(Jacobian) at all Gauss quadrature points"""
        # get the nodal locations weights and shape functions for a given GLJ quadrature
        ksis, weights, NN, NN_r, JJ = \
                            shape_fun.line_spectral_GLJ_lagrange(self.nodes_at)

        # extract the matrices and sum at each integration point
        for i, ksi in enumerate(list(ksis)):

            # shape funcion values at current location are NN[:, i]
            # generate the shape function matrix in standard form
            N = NN[:, i].reshape(1, -1)
            N_r = NN_r[:, i].reshape(1, -1)

            # local r in physcial coordinates
            r_i = self.nodes_at.dot(NN[:, i])
            # scale the radius according to the de'l'Hospital's rule for the axis
            # singularity

            w = 1 + ksi
            if np.round(r_i, 7) == 0:
                rat = JJ[i]
            else:
                rat = r_i/w
            # use different forms for the axis of symmetri and the rest of the domain
            if np.round(r_i, 7) == 0:
                K_5 = N_r.T.dot(N_r)
            else:
                K_5 = 1/r_i**2*N.T.dot(N)

            K_1 = N_r.T.dot(N_r)
            K_6 = N.T.dot(N)
            M = N.T.dot(N)/self.c_p**2
            # GLJ integration
            self.K_1 += -self.rho*2*np.pi*weights[i]*K_1*JJ[i]*rat
            self.K_5 += -self.rho*2*np.pi*weights[i]*K_5*JJ[i]*rat
            self.K_6 += -self.rho*2*np.pi*weights[i]*K_6*JJ[i]*rat
            self.K_Fz += -self.rho*2*np.pi*weights[i]*K_6*JJ[i]*rat
            self.K_Ft += -self.rho*2*np.pi*weights[i]*K_5*JJ[i]*rat
            self.M += -self.rho*2*np.pi*weights[i]*M*JJ[i]*rat

            # if the element is coupled to solid elements - create the ingredients
            # of the coupling matrices
            N_phi = NN[:, -1].reshape(-1, 1)
            self.Hf1 = self.rho*N_phi*self.nodes_at[-1]
        # apply the boundary conditions at the axis of symmetry
        if (0 in self.nodes_at) & (self.n != 0):
            self.K_1 = np.delete(np.delete(self.K_1, [0], 0), [0], 1)
            self.K_5 = np.delete(np.delete(self.K_5, [0], 0), [0], 1)
            self.K_6 = np.delete(np.delete(self.K_6, [0], 0), [0], 1)
            self.M = np.delete(np.delete(self.M, [0], 0), [0], 1)
            self.K_F = np.delete(np.delete(self.K_F, [0], 0), [0], 1)
            self.K_Ft = np.delete(np.delete(self.K_Ft, [0], 0), [0], 1)
            self.K_Fz = np.delete(np.delete(self.K_Fz, [0], 0), [0], 1)


class SLAX6_PML(object):
    """
    Class for an axisymmetric PML 1D line element with nodes wrt GLL quadrature
    and higher order Lagrange interpolating polynomials
    u = [ur utheta uz]
    strain directions = [rr tt zz tz rz rt]
    n is the circumferential order number"""
    def __init__(self, nodes_at, n, PML_params, PML_function=gamma_PML_exp):
        """
        Initialise the element with node locations and PML parameters

        Parameters:
            nodes_at : node locations
            PML_params : parameters of the PML as a list [start, thickness, a + 1j*b]
            PML_function : function defining the PML profile (exponential by default)
        """

        self.nodes_at = nodes_at
        self.nodes_per_el = len(nodes_at)
        self.no_of_dofs = 3*self.nodes_per_el
        # specify basic element characteristics and T-transformation
        self.dofs_per_node = [3]*self.nodes_per_el
        self.dofs_domain = [['s']*3]*self.nodes_per_el
        self.dofs_per_el = sum(self.dofs_per_node)
        self.T_components = [[1, 1j, 1j]]*self.nodes_per_el
        self.n = n

        self.C = None
        self.loss_factor = None
        self.rho = None

        # preallocate the arrays
        self.K_Fz = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_Ft = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_F = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_1 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_2 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_3 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_4 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_5 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.K_6 = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')
        self.M = np.zeros([self.no_of_dofs, self.no_of_dofs], dtype='complex')

        self.Hs0 = None
        self.PML_start = PML_params[0]
        self.PML_thk = PML_params[1]
        self.gamma = PML_params[2]
        self.PML_function = PML_function

    def add_properties(self, list_of_props):
        """Assign mechanical properties and the circumferential order.

        Parameters:
            list_of_props : [lame_1, lame_2, denisty, loss_factor]
            n : circumferential order
        """
        # assign the properties
        self.loss_factor = list_of_props[3]
        lame_1 = list_of_props[0]
        lame_2 = list_of_props[1]
        self.C = (1 + self.loss_factor*1.0j)*\
                np.array([(lame_1 + 2*lame_2, lame_1, lame_1, 0, 0, 0),
                          (lame_1, lame_1 + 2*lame_2, lame_1, 0, 0, 0),
                          (lame_1, lame_1, lame_1 + 2*lame_2, 0, 0, 0),
                          (0, 0, 0, lame_2, 0, 0),
                          (0, 0, 0, 0, lame_2, 0),
                          (0, 0, 0, 0, 0, lame_2)])
        self.rho = list_of_props[2]


    def calculate_matrices(self):
        """Calculates the shape function and b matrices including
        Jacobian*Gauss Weight* det(Jacobian) at all Gauss quadrature points"""
        # get the nodal locations weights and shape functions for a given GLJ quadrature
        ksis, weights, NN, NN_r, JJ = \
                            shape_fun.line_spectral_GLL_lagrange(self.nodes_at)

        # extract the matrices and sum at each integration point
        for i in range(len(ksis)):
            # shape funcion values at current location are NN[:, i]
            # generate the shape function matrix in standard form
            diagonals = [[funct]*3 for funct in NN[:, i]]
            N = np.column_stack([np.diag(diagonal) for diagonal in diagonals])

            diagonals_r = [[funct]*3 for funct in NN_r[:, i]]
            N_r = np.column_stack([np.diag(diagonal) for diagonal in diagonals_r])

            # local r in physical coordinates
            r_i = self.nodes_at.dot(NN[:, i])
            # calculate the stretched r
            if self.PML_function is gamma_PML:
                r_tilde = -(self.gamma - 1)*(self.PML_start - r_i)**3/self.PML_thk**2 + r_i
            elif self.PML_function is gamma_PML_exp:
                local_r = (r_i - self.PML_start)/self.PML_thk
                r_tilde = self.PML_start + self.PML_thk/self.gamma.real*(
                    np.exp(self.gamma.real*local_r) - 1) - \
                        1j*self.PML_thk/self.gamma.imag*(
                            np.exp(self.gamma.imag*local_r) - 1) + 1j*local_r*self.PML_thk
            else:
                print('Error. No such PML function defined')

            # calculate the value of the PML profile at current integration point
            current_gamma = self.PML_function(r_i, self.gamma, \
                                  self.PML_start, self.PML_thk)
            # differential operator matrices
            L_z = np.zeros([6, 3])
            L_theta = np.zeros([6, 3])
            L_r = np.zeros([6, 3])
            L = np.zeros([6, 3])

            L_z[2, 2], L_z[3, 1], L_z[4, 0] = 1, 1, 1
            L_r[0, 0], L_r[4, 2], L_r[5, 1] = 1, 1, 1
            L[1, 0], L[5, 1] = 1.0, -1.0
            L_theta[1, 1], L_theta[3, 2], L_theta[5, 0] = 1, 1, 1

            B_1 = L.dot(N)/r_tilde + 1/current_gamma*L_r.dot(N_r)
            B_2 = L_theta.dot(N)/r_tilde
            B_3 = L_z.dot(N)

            K_F1 = B_1.T.dot(self.C).dot(B_3)
            K_F2 = B_1.T.dot(self.C).dot(B_2)
            K_F3 = B_2.T.dot(self.C).dot(B_3)

            K_1 = B_1.T.dot(self.C).dot(B_1)
            K_2 = K_F2.T - K_F2
            K_3 = K_F1.T - K_F1
            K_4 = K_F3.T + K_F3
            K_5 = B_2.T.dot(self.C).dot(B_2)
            K_6 = B_3.T.dot(self.C).dot(B_3)

            K_F = K_F1.T
            K_Ft = K_F3.T
            K_Fz = K_6

            M = self.rho*N.T.dot(N)
            # GLL integration
            self.K_F += 2*np.pi*weights[i]*K_F*JJ[i]*r_tilde*current_gamma
            self.K_Ft += 2*np.pi*weights[i]*K_Ft*JJ[i]*r_tilde*current_gamma
            self.K_Fz += 2*np.pi*weights[i]*K_Fz*JJ[i]*r_tilde*current_gamma

            self.K_1 += 2*np.pi*weights[i]*K_1*JJ[i]*r_tilde*current_gamma
            self.K_2 += 2*np.pi*weights[i]*K_2*JJ[i]*r_tilde*current_gamma
            self.K_3 += 2*np.pi*weights[i]*K_3*JJ[i]*r_tilde*current_gamma
            self.K_4 += 2*np.pi*weights[i]*K_4*JJ[i]*r_tilde*current_gamma
            self.K_5 += 2*np.pi*weights[i]*K_5*JJ[i]*r_tilde*current_gamma
            self.K_6 += 2*np.pi*weights[i]*K_6*JJ[i]*r_tilde*current_gamma

            self.M += 2*np.pi*weights[i]*M*JJ[i]*r_tilde*current_gamma

        # if the element is coupled to fluid elements - create the ingredients
        # of the coupling matrices
        normal = np.array([1, 0, 0])
        diagonals = [[funct]*3 for funct in NN[:, 0]]
        N_u = np.column_stack([np.diag(diagonal) for diagonal in diagonals])
        self.Hs0 = 2*np.pi*N_u.T.dot(normal.reshape(-1, 1))

class ALAX6_PML(object):
    """
    Class for an acoustic axisymmetric 1D line element with nodes wrt GLL quadrature
    and higher order Lagrange interpolating polynomials belonging to a PML
    each node has one degree of freedom - velocity potential
    n is the circumferential order number"""
    def __init__(self, nodes_at, n, PML_params, PML_function=gamma_PML_exp):
        """
        Initialise the element with node locations and PML parameters

        Parameters:
            nodes_at : node locations
            PML_params : parameters of the PML as a list [start, thickness, a + 1j*b]
            PML_function : function defining the PML profile (exponential by default)
        """
        self.nodes_at = nodes_at
        self.nodes_per_el = len(nodes_at)
        self.no_of_dofs = self.nodes_per_el
        # define basic properties of the element and T-transformation
        self.dofs_per_node = [1]*self.nodes_per_el
        self.dofs_domain = [['f']]*self.nodes_per_el
        self.dofs_per_el = sum(self.dofs_per_node)
        self.T_components = [[1]]*self.nodes_per_el
        self.n = n

        self.c_p = None
        self.rho = None
        # preallocate the arrays
        self.K_Fz = np.zeros([self.no_of_dofs, self.no_of_dofs],
                             dtype='complex')
        self.K_Ft = np.zeros([self.no_of_dofs, self.no_of_dofs],
                             dtype='complex')
        self.K_F = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex')
        self.K_1 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dz*dz
        self.K_2 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dtheta*dz
        self.K_3 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dz
        self.K_4 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dtheta*dtheta
        self.K_5 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # dtheta
        self.K_6 = np.zeros([self.no_of_dofs, self.no_of_dofs],
                            dtype='complex') # -
        self.M = np.zeros([self.no_of_dofs, self.no_of_dofs],
                          dtype='complex')
        self.Hf0 = None

        self.PML_start = PML_params[0]
        self.PML_thk = PML_params[1]
        self.gamma = PML_params[2]
        self.PML_function = PML_function

    def add_properties(self, list_of_props):
        """Assign mechanical properties.

        Parameters:
            list_of_props : [bulk modulus, density]

        """
        # assign properties
        self.c_p = (list_of_props[0]/list_of_props[1])**0.5
        self.rho = list_of_props[1]

    def calculate_matrices(self):
        """Calculates the shape function and b matrices including
        Jacobian*Gauss Weight* det(Jacobian) at all Gauss quadrature points"""
        # get the nodal locations weights and shape functions for a given GLL quadrature
        ksis, weights, NN, NN_r, JJ = \
                            shape_fun.line_spectral_GLL_lagrange(self.nodes_at)

        # extract the matrices and sum at each integration point
        for i in range(len(ksis)):

            # shape funcion values at current location are NN[:, i]
            # generate the shape function matrix in standard form
            N = NN[:, i].reshape(1, -1)
            N_r = NN_r[:, i].reshape(1, -1)

            # local r in physical coordinates
            r_i = self.nodes_at.dot(NN[:, i])
            # calculate the stretched r according to the chosen PML profile
            if self.PML_function is gamma_PML:
                r_tilde = -(self.gamma - 1)*(self.PML_start - r_i)**3/self.PML_thk**2 + r_i
            elif self.PML_function is gamma_PML_exp:
                local_r = (r_i - self.PML_start)/self.PML_thk
                r_tilde = self.PML_start + self.PML_thk/self.gamma.real*(
                    np.exp(self.gamma.real*local_r) - 1) - \
                            1j*self.PML_thk/self.gamma.imag*(
                                np.exp(self.gamma.imag*local_r) - 1) + 1j*local_r*self.PML_thk
            else:
                print('Error. No such PML function defined')
            # calculate the value of the PML profile at the current location
            current_gamma = self.PML_function(r_i, self.gamma, \
                                  self.PML_start, self.PML_thk)
            K_1 = 1/current_gamma**2*N_r.T.dot(N_r)
            K_5 = 1/r_tilde**2*N.T.dot(N)
            K_6 = N.T.dot(N)
            M = N.T.dot(N)/self.c_p**2
            # GLL integration
            self.K_6 += -self.rho*2*np.pi*weights[i]*K_6*JJ[i]*r_tilde*current_gamma
            self.K_1 += -self.rho*2*np.pi*weights[i]*K_1*JJ[i]*r_tilde*current_gamma
            self.K_5 += -self.rho*2*np.pi*weights[i]*K_5*JJ[i]*r_tilde*current_gamma
            self.K_Fz += -self.rho*2*np.pi*weights[i]*K_6*JJ[i]*r_tilde*current_gamma
            self.K_Ft += -self.rho*2*np.pi*weights[i]*K_5*JJ[i]*r_tilde*current_gamma
            self.M += -self.rho*2*np.pi*weights[i]*M*JJ[i]*r_tilde*current_gamma
        # if the element is coupled to solid elements - create the ingredients
        # of the coupling matrices
        N_phi = NN[:, 0].reshape(-1, 1)
        self.Hf0 = self.rho*N_phi
