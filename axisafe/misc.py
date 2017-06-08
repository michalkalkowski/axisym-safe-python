"""
==============================================================================
 Copyright (C) 2013--2014 Michal Kalkowski
 kalkowski.m@gmail.com

 This file contains some miscellaneous functions for SAFE/wave approach implemented in Python.
 Some parts are not very intuitive, but it is all at the development stage.
 ==============================================================================
"""

from __future__ import print_function
import numpy as np


def G_nu2lame(G, nu):
    """
    Converts shear modulus and Poisson's ratio to Lame constants.

    Parameters:
        G : shear modulus
        nu : Poisson's ratio

    Returns:
        lame1 : lambda
        lame2 : mu
    """
    return 2*G*nu/(1 - 2*nu), G

def bulkvel2E(cl, ct, rho):
    """
    Converts bulk wave velocities to Young's modulus and Poisson's ratio.

    Parameters:
        cl : longitudinal wave velocity
        ct : shear wave velocity
        rho : density

    Returns:
        E : Young's modulus
        nu : Poisson's ratio
    """
    lame_2 = ct**2*rho
    nu = (2 - cl**2/ct**2)/(2*(1 - cl**2/ct**2))
    E = lame_2*2*(1 + nu)
    return E, nu

def cLcS2lame(cl, ct, rho):
    """
    Converts bulk wave velocities to Lame constants.

    Parameters:
        cl : longitudinal wave velocity
        ct : shear wave velocity
        rho : density

    Returns:
        lame1 : lambda
        lame2 : mu
    """
    lame_1 = (cl**2 - 2*ct**2)*rho
    lame_2 = ct**2*rho
    return lame_1, lame_2

def young2lame(E, nu):
    """
    Converts Young's modulus and Poisson's ratio to Lame constants.

    Parameters:
        E : Young's modulus
        nu : Poisson's ratio

    Returns:
        lame1 : lambda
        lame2 : mu
    """
    lame_1 = nu*E/(1 + nu)/(1 - 2*nu)
    lame_2 = E/2/(1 + nu)

    return lame_1, lame_2

def young2C(E, nu):
    """
    Assmbles the stiffness matrix C based on the Young's modulus and Poisson's ratio.

    Parameters:
        E : Young's modulus
        nu : Poisson's ratio

    Returns:
        C : stiffness matrix
    """
    lame_1, lame_2 = young2lame(E, nu)
    C = np.zeros([6, 6], 'complex')
    C[0, 0] = lame_1 + 2*lame_2
    C[0, 1], C[0, 2] = lame_1, lame_1
    C[1, 0], C[1, 2] = lame_1, lame_1
    C[1, 1] = C[0, 0]
    C[2, 0], C[2, 1] = lame_1, lame_1
    C[2, 2] = C[0, 0]
    C[3, 3], C[4, 4], C[5, 5] = lame_2, lame_2, lame_2
    return C

def lame2C(lame_1, lame_2):
    """
    Assmbles the stiffness matrix C based on the Lame constants.

    Parameters:
        lame1 : lambda
        lame2 : mu

    Returns:
        C : stiffness matrix
    """
    C = np.zeros([6, 6], 'complex')
    C[0, 0] = lame_1 + 2*lame_2
    C[0, 1], C[0, 2] = lame_1, lame_1
    C[1, 0], C[1, 2] = lame_1, lame_1
    C[1, 1] = C[0, 0]
    C[2, 0], C[2, 1] = lame_1, lame_1
    C[2, 2] = C[0, 0]
    C[3, 3], C[4, 4], C[5, 5] = lame_2, lame_2, lame_2
    return C
