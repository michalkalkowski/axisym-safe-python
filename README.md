                            Axisymmetric semi-analytical finite elements in Python

What is it?
===========

This repository contains the axisafe package which is an implementation of 
the axisymmetric finite elements in Python, following a publication in 
Computers & Structures. The code can be used to find mechanical waves
propagating axisymmetric structures, such as pipes. These can be free, buried
or submerged, and may contain a fluid inside. An arbitrary composition of layers
is allowed. 

A few examples corresponding to the cases considered in the aforementioned paper
are also included.

The code was written ant tested for:
* Python 3.5
* numpy 1.12.1
* scipy 0.19.0
* matplotlib 2.0.2

Whilst it should work fine with older versions as well, plotting code including in the example scripts may not execute owing to changes in color definition in matplotlib. This does not compromise the core functionality of the package.

License
=======

All course materials are licensed under the MIT License:

> Copyright (c) 2017 Michał Kalkowski

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:

> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.

> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
> THE SOFTWARE.

Publication
=========

Full bibliographic details of the parent publication are:

Kalkowski MK et al. **Axisymmetric semi-analytical finite elements for modelling waves in buried/submerged fluid-filled waveguides**. *Comput Struct* (2017), https://doi.org/10.1016/j.compstruc.2017.10.004

Whenever you use this code, please cite the above mentioned paper as an attribution to the authors.

Erratum
=========
An ambiguity in the fluid element equations in the published version of the paper became apparent. Eq. (35) ommits the $\rho_f$ factor that multiplies both terms in the virtual work equation. This ommision may be not recalled unless one considers structural acoustic coupling. The code has always preserved the dimensions appropriately. 

I no short, all matrices related to acoustic elements (Eq. (24)) should be multiplied by $\rho_f$, which would be best done at element integration stage. Practically, this means adding the $\rho_f$ factor to all matrices in Eq. (24). 

As mentioned this is an ommision in the printed version of the equations and not in the implementation.

Request
=========
Please note that this software is written by a mechanical engineer, not a programmer, and is far from ideal from the software engineering perspective. For this reason, it is very much open to improvement and any contribution is more than welcome.
