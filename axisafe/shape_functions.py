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

This file contains element shape functions definitions.
 ==============================================================================
"""
from math import factorial as fac
import numpy as np
from scipy.special import gamma

def jacobi(x, order, alpha, beta):
    """
    Calculates Jacobi polynomials of a given order, alpha and beta for x.
    See Karniadakis and Sherwin (1999) for details.
    
    Parameters:
        x : array, argument
        order : int, order
        alpha : int
        beta : int

    Rerutns:
        P[:, -1] : Jacobi polynomials of a given order, alpha, beta at x
    """
    # check if the calculation is for a point or an array
    if type(x) is np.ndarray:
        P = np.zeros([len(x), order + 1])
    else:
        P = np.zeros([1, order + 1])
    # the first two are known (closed-form)
    P[:, 0] = 1
    P[:, 1] = 0.5*(alpha - beta + (alpha + beta + 2)*x)
    # Jacobi polynomials - recursive formulae
    for n in range(1, order):
        # Eq. (A.3) in Karniadakis and Spencer, 1999
        a1 = 2*(n + 1)*(n + alpha + beta + 1)*(2*n + alpha + beta)
        a2 = (2*n + alpha + beta + 1)*(alpha**2 - beta**2)
        a3 = (2*n + alpha + beta)*(2*n + alpha + beta + 1)*(2*n + alpha + beta + 2)
        a4 = 2*(n + alpha)*(n + beta)*(2*n + alpha + beta + 2)
        P[:, n + 1] = ((a2 + a3*x)*P[:, n] - a4*P[:, n - 1])/a1
    return P[:, -1]

def djacobi(x, order, alpha, beta):
    """
    Calculates the derivative of Jacobi polynomials.
    Parameters:
        x : array, argument
        order : int, order
        alpha : int
        beta : int

    Rerutns:
        dP : derivatives of Jacobi polynomials of a given order, alpha, beta at x
        
    """
    dP = 0.5*(alpha + beta + order + 1)*\
            jacobi(x, order - 1, alpha + 1, beta + 1)
    return dP

def jacobi_roots(order, alpha, beta):
    """
    Calculates roots of Jacobi polynomials given order, alpha and beta.
    
    Parameters:
        order : int, order
        alpha : int, alpha
        beta : int, beta

    Returns:
        x : array, roots of the given Jacobi polynomial.
    """
    eps = np.finfo(float).eps
    x = np.zeros(order)
    for k in range(order):
        # initial guess - zeros of Chebychev polynomials (alpha, beta = -0.5)
        r = - np.cos(np.pi*(2*k + 1)/(2*order))
        if k > 0:
            r = (r + x[k - 1])/2
        delta = 1
        while np.max(abs(delta)) > eps:
            s = 0
            # polynomial deflation - eliminating identified roots
            for i in range(k):
                s += 1/(r - x[i])
            P = jacobi(r, order, alpha, beta)
            dP = djacobi(r, order, alpha, beta)
            delta = -P/(dP- s*P)
            r += delta
        x[k] = r
    return x

def build_GLL_quadrature(no_of_nodes):
    """
    Constructs Gauss-Lobatto-Legendre nodes and weights. Nodes are located at
    the roots of (1- ksi^2)*Pn'(ksi) = 0 where Pn' is a Legendre polynomial
    of order n (note that n=no_of_nodes - 1) and ksi is the local coordonate
    between -1 and 1.
    This function is based on Greg von Winckel's MATLAB code
    https://uk.mathworks.com/matlabcentral/fileexchange/4775-legende-gauss-lobatto-nodes-and-weights/content/lglnodes.m
    and John Burkhardt's implementation

    Parameters:
        no_of_nodes : int, the number of nodes (order + 1)
    
    Returns:
        ksi : array, GLL node locations
        w : array, integration weights
    """

    order = no_of_nodes - 1
    # first guess at Chebyshev-Gauss-Lobatto nodes
    ksi = np.cos(np.pi*np.arange(order + 1)/(order))

    # Legendre Vandermonde matrix
    P = np.zeros([order + 1, order + 1])

    # Pn is computed using a recursive relation.
    xold = 2
    eps = np.finfo(float).eps

    while np.max(abs(ksi - xold)) > eps:
        xold = ksi
        P[:, 0] = 1
        P[:, 1] = ksi
        for k in range(2, order + 1):
            P[:, k] = ((2*k - 1)*ksi*P[:, k - 1] - (k - 1)*P[:, k - 2])/k
        ksi = xold - (ksi*P[:, order] - P[:, order - 1])/(order*P[:, order])
    w = 2/(order*(order + 1)*P[:, order]**2)
    ksi.sort()

    return ksi, w

def build_GLJ_quadrature(Q):
    """
    Constructs Gauss-Lobatto-Jacobi (0, 1) nodes and weights.
    Following Karniakadis and Sherwin 1999 Appendix B.2
    and https://github.com/pjabardo/Jacobi.j
    
    Parameters:
        Q : int, number of nodes

    Returns:
        ksi : array, node locations
        w : array, integration weights    
    """
    alpha, beta = 0, 1
    interior = jacobi_roots(Q - 2, alpha + 1, beta + 1)
    ksi = np.array([-1] + list(interior) + [1])

    # Following KArniadakis and Sherwin, 1999
    # Appendix B, p.356
    num = 2**(alpha + beta + 1)*gamma(alpha + Q)*gamma(beta + Q)
    den = (Q - 1)*fac(Q - 1)*gamma(alpha + beta + Q + 1)
    C_coeff = num/den
    w = np.zeros(Q)
    w[0] = (beta + 1)*C_coeff/(jacobi(ksi[0], Q - 1, alpha, beta))**2
    w[-1] = (alpha + 1)*C_coeff/(jacobi(ksi[-1], Q - 1, alpha, beta))**2
    for i in range(1, Q - 1):
        w[i] = C_coeff/(jacobi(ksi[i], Q - 1, alpha, beta))**2
    return ksi, w

def lagrange_poly(ksi, order=0):
    """
    Calculates the values and the  derivatives of the Lagrange polynomials over the GLL nodes.
    Derivatives calculated based on 
    http://math.stackexchange.com/questions/1105160/evaluate-derivative-of-lagrange-polynomials-at-construction-points
 
    Parameters:
        ksi : array, node locations
        order: int, Lagrange polynomial order

    Returns:
        N_nodal : array, values of the Lagrange interpolating polynomials
        dN_nodal : array, derivatives of the Lagrange interpolating polynomials
    """
    if order == 0:
        order = len(ksi)
    N_nodal = np.zeros([order, len(ksi)])
    dN_nodal = np.zeros([order, len(ksi)])
    for i in range(order):
        temp_nodal = 1
        for j in range(order):
            if i != j:
                temp_nodal *= (ksi - ksi[j])/(ksi[i] - ksi[j])
        N_nodal[i] = temp_nodal
    for i in range(order):
        temp_nodal = 0
        for j in range(len(ksi)):
            if j != i:
                s1_nodal = 1/(ksi[i] - ksi[j])
                for k in range(len(ksi)):
                    if k != i and k != j:
                        s1_nodal *= (ksi - ksi[k])/(ksi[i] - ksi[k])
                temp_nodal += s1_nodal
        dN_nodal[i] = temp_nodal
    return N_nodal, dN_nodal

def lagrange_GLJ(ksi, Q=0):
    """
    Calculates the values and the  derivatives of the Lagrange polynomials over the GLJ (0, 1) nodes.
 
    Parameters:
        ksi : array, node locations
        Q: int, Lagrange polynomial order

    Returns:
        N_nodal : array, values of the Lagrange interpolating polynomials
        dN_nodal : array, derivatives of the Lagrange interpolating polynomials
    Lagrange shape functions and its derivatives over GLJ nodes
    """
    if Q == 0:
        Q = len(ksi)
    ksij, _ = build_GLJ_quadrature(Q)

    # interpolation order
    N = Q - 1
    Pj = jacobi(ksij, N, 0, 1)
    dP = djacobi(ksi, N, 0, 1)
    N_nodal = np.zeros([len(ksi), Q])
    N_nodal[:, 0] = 2*(-1)**N*(ksi - 1)*dP/((N + 1)*N*(N + 2))
    N_nodal[:, -1] = (1 + ksi)*dP/N/(N + 2)
    for i in range(1, N):
        for j in range(N):
            if i == j:
                N_nodal[j, i] = 1
            else:
                N_nodal[j, i] = (ksi[j]**2 - 1)*dP[j]/(N*(N + 2)*Pj[i]*(ksi[j] - ksij[i]))

    # Follwing T. Nissen-Mayer A Fournier, et al.
    dN_nodal = np.zeros([len(ksij), Q])
    for I in range(len(ksij)):
        for i in range(Q):
            if i == 0 and I == 0:
                dN_nodal[i, I] = -N*(N + 2)/6
            elif i == 0 and 1 <= I <= N - 1:
                dN_nodal[i, I] = 2*(-1)**N*jacobi(ksij[I], N, 0, 1)/\
                            (1 + ksij[I])/(N + 1)
            elif i == 0 and I == N:
                dN_nodal[i, I] = (-1)**N/(N + 1)
            elif 1 <= i <= N - 1 and I == 0:
                dN_nodal[i, I] = (-1)**Q*Q/(2*jacobi(ksij[i], N, 0, 1)*(1 + ksij[i]))
            elif 1 <= i <= N - 1 and 1 <= I <= N - 1 and i != I:
                dN_nodal[i, I] = 1/(ksij[I] - ksi[i])*jacobi(ksij[I], N, 0, 1)/\
                            jacobi(ksij[i], N, 0, 1)
            elif 1 <= i <= N - 1 and I == i:
                dN_nodal[i, I] = -1/(2*(1 + ksij[i]))
            elif 1 <= i <= N - 1 and I == N:
                dN_nodal[i, I] = 1/jacobi(ksij[i], N, 0, 1)/(1 - ksij[i])
            elif i == N and I == 0:
                dN_nodal[i, I] = (-1)**(N + 1)*(N + 1)/4
            elif i == N and 1 <= I <= N - 1:
                dN_nodal[i, I] = -jacobi(ksij[I], N, 0, 1)/(1 - ksij[I])
            elif i == N and I == N:
                dN_nodal[i, I] = (N*(N + 2) - 1)/4
    return N_nodal, dN_nodal

def line_spectral_GLL_lagrange(nodes):
    """
    Returns node locations, weights, interpolants and the Jacobian for
    a line spectral element based on GLL nodes.
    
    Paramters:
        nodes : array, nodal locations
    
    Returns:
        ksi : array, GLL node locations
        weights : array, integration weights
        N_nodal : array, interpolants at the nodes
        N_r : array, the derivatives of the interpolants at the nodes
        J : float, the determinant of the Jacobian.
    """
    # infer the order from node locations
    ksi, weights = build_GLL_quadrature(len(nodes))
    # calculate shape functions and derivatives in natural coordinates
    N_nodal, dN_nodal = lagrange_poly(ksi)
    # calculate Jacobians for each ksi location
    J = dN_nodal.T.dot(nodes)
    # calculate derivatives of shape functions in physical coordinates
    N_r = 1/J*dN_nodal

    return ksi, weights, N_nodal, N_r, J

def line_spectral_GLJ_lagrange(nodes):
    """
    Returns node locations, weights, interpolants and the Jacobian for
    a line spectral element based on GLJ (0, 1)  nodes.
    
    Paramters:
        nodes : array, nodal locations
    
    Returns:
        ksi : array, GLL node locations
        weights : array, integration weights
        N_nodal : array, interpolants at the nodes
        N_r : array, the derivatives of the interpolants at the nodes
        J : float, the determinant of the Jacobian.    
    """
    # infer the order from node locations
    ksi, weights = build_GLJ_quadrature(len(nodes))
    N_nodal, dN_nodal = lagrange_GLJ(ksi)
    # calculate Jacobians for each ksi location
    J = nodes.dot(dN_nodal)
    # calculate derivatives of shape functions in physical coordinates
    N_r = 1/J*dN_nodal

    return ksi, weights, N_nodal, N_r, J
