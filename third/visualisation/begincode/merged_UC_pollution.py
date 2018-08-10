"""
concentration part

"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

T = 10.0           # final time
num_steps = 100    # number of time steps
dt = T / num_steps # time step size
eps = 1.00         # we will take this to zero
nu = 0.01          # viscosity, later set to (nu_1, nu_2, nu_3)
K1 = 1             # diffusion in the x direction
K2 = 1             # diffusion in the y direction
K3 = 1             # diffusion in the z direction
s = 1              # source term function, constant for now

# Read mesh from file
mesh = Mesh('merged_UC_eqs/mesh.xml.gz')

# Define function space for velocity
U = VectorFunctionSpace(mesh, 'P', 2)

# Define function space for system of concentrations
C = FunctionSpace(mesh, 'P', 1)

# Define test function for pollution concentration
d = TestFunction(C)

# Define functions for velocity and concentrations
u = Function(U)
c = Function(C)
c_n = Function(C)

# Define expressions used in variational forms
k = Constant(dt)
eps = Constant(eps)
K1 = Constant(K1)
K2 = Constant(K2)
K3 = Constant(K3)

# Define variational problem
F = ((c - c_n)/k) * d * dx \
  - dot(u * c, grad(d)) * dx \
  + eps * K1 * c.dx(0) * d.dx(0) * dx \
  + K2 * c.dy(0) * d.dy(0) * dx \
  + K3 * c.dz(0) * d.dz(0) * dx \
  - s * d * dx

# Create time series for reading velocity data
timeseries_u = TimeSeries('merged_UC_eqs/velocity_series')

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Read velocity from file
    timeseries_u.retrieve(u.vector(), t)

    # Solve variational problem for time step
    solve(F == 0, c)

    # Update previous solution
    c_n.assign(c)
