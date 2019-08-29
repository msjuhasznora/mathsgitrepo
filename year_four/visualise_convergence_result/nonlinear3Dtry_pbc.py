# periodic boundary conditions, no boundary integral term.
# it converges immediately to the constant (0,0,0) function as that actually is a solution.

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import datetime
from numpy.random import rand

from dolfin import *
import numpy

class PeriodicBoundary(SubDomain):
    
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        if bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary):
            return True
        if bool(x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS and on_boundary):
            return True
        if bool(x[2] < DOLFIN_EPS and x[2] > -DOLFIN_EPS and on_boundary):
            return True

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        if near(x[0], 1):
            y[0] = x[0] - 1
            y[1] = x[1]
            y[2] = x[2]
        if near(x[1], 1):
            y[0] = x[0]
            y[1] = x[1] - 1
            y[2] = x[2]
        if near(x[2], 1):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - 1


# Create mesh and define function space
mesh = UnitCubeMesh(20, 20, 20)
V = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 3, constrained_domain=PeriodicBoundary())

# Choice of nonlinear coefficient

eps = 1.0

# Define variational problem
v = TestFunction(V)
u = Function(V)
u1, u2, u3 = split(u)
v1, v2, v3 = split(v)
F = inner(u, grad(u1)) * v1 * dx + inner(u, grad(u2)) * v2 * dx + eps * eps * inner(u, grad(u3)) * v3 * dx + \
inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx + eps * eps * inner(grad(u3),grad(v3)) * dx

# Compute solution
solve(F == 0, u, solver_parameters = {"newton_solver":{ "linear_solver" : "mumps"}})

file = File("nonlinear_u.pvd")
file << u
