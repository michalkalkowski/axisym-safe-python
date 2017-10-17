"""
 ==============================================================================
 Copyright (C) 2016--2017 Michal Kalkowski (MIT License)
 kalkowski.m@gmail.com

 This is an init file for the axisafe package developed for simulating elastic
 wave propagation in buried/submerged fluid-filled waveguides. The package is 
 based on the publication:
     Kalkowski MK et al. Axisymmetric semi-analytical finite elements for modelling 
     waves in buried/submerged fluid-filled waveguides. Comput Struct (2017), 
     https://doi.org/10.1016/j.compstruc.2017.10.004

 ==============================================================================
"""

from . import mesh
from . import misc
from . import solver

__all__ = ['mesh', 'elements', 'shape_functions', 'solver', 'misc']
