"""
velocity part

"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

# Define constants
T = 10.0           # final time
num_steps = 100    # number of time steps
dt = T / num_steps # time step size
eps = 1         # we will take this to zero
nu = 1          # viscosity, later set to (nu_1, nu_2, nu_3)
alpha = 1        # alpha
beta = 1         # beta

print("Constants defined.")

# Create mesh
mesh = UnitCubeMesh(16, 16, 16)

print("Mesh created.")

P1 = VectorElement('Lagrange', tetrahedron, 2, dim = 2) # element for the horizontal velocity
P2 = FiniteElement('Lagrange', tetrahedron, 2) # element for the vertical velocity
element = MixedElement([P1, P2]) # mixed element for the 3D velocity
V = FunctionSpace(mesh, element)

print("Function space created.")

# Define test functions
v_h, v_3 = TestFunctions(V)
v_1, v_2 = split(v_h)

print("Test functions defined.")

# Define functions for velocity
u = Function(V)
u_n = Function(V)

# Split system functions to access components
u_h, u_3 = split(u)
u_1, u_2 = split(u_h)
u_hn, u_3n = split(u_n)

# Define expressions used in variational forms
k = Constant(dt)
nu = Constant(nu)
eps = Constant(eps)

# Define variational problem
F = dot((u_h - u_hn)/k, v_h) * dx \
  - dot(u_1, dot(u, grad(v_1))) * dx \
  - dot(u_2, dot(u, grad(v_2))) * dx \
  + dot(alpha * (- u_2, u_1), v_h) * dx \
  + dot(nu * grad(u_1), nu * grad(v_1)) * dx \
  + dot(nu * grad(u_2), nu * grad(v_2)) * dx \
  + eps^2 * dot((u_3 - u_3n)/k, v_3) * dx \
  + eps^2 * dot(dot(u, grad(u_3)), v_3) * dx \
  + eps^2 * dot(nu * u_3, nu * v_3) * dx \
  + eps *  dot(beta * u_3, v_1) * dx \
  - eps * dot(beta * u_1 , v_3) * dx

print("Variational problem defined.")

timeseries_u = TimeSeries('merged_UC_eqs/velocity_series')
File('merged_UC_eqs/mesh.xml.gz') << mesh

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Solve variational problem for time step
    solve(F == 0, u)

    timeseries_u.store(u.vector(), t)

    # Update previous solution
    u_n.assign(u)
