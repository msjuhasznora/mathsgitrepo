# nonlinear solver, boundary integral included, the Jacobian is computed automatically.
# gmres solver
# stationary
# this version contains the "strong form" of the div-free condition, ie we use u1 * grad(v1) instead of grad(u1) * v1, and taking implicitely advantage that the divergence is zero. however this is not explicitely told to the program, so it does not really know, nor in a weak, nor in a strong form. so this version is missing the P pressure space and some explicit form of the incompressibility condition.

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import datetime
from numpy.random import rand
from ufl import div
from dolfin import *
import numpy

# Create mesh and define function space
mesh = UnitCubeMesh(20, 20, 20)
V = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 3)

# Define boundary conditions
class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 1.0)
upperboundary = UpperBoundary()
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
upperboundary.mark(boundaries, 1)
ds = Measure('ds')[boundaries]

noslipbasin = DirichletBC(V, (0, 0, 0), "on_boundary && x[2] < 1.0 - DOLFIN_EPS")
zerotop = DirichletBC(V.sub(2), 0, "on_boundary && x[2] > 1.0 - DOLFIN_EPS")

bcu = [noslipbasin, zerotop]
#bcu = [noslipbasin]

wind_shear_x = 60.0
wind_shear_y = 30.0
theta = Constant((wind_shear_x, wind_shear_y, 0))

# Choice of nonlinear coefficient

f1 = 20.0
f2 = 20.0
f3 = 20.0
eps = 1.0

# Define variational problem
v = TestFunction(V)
u = Function(V)
u1, u2, u3 = split(u)
v1, v2, v3 = split(v)
F = - inner(u, grad(v1)) * u1 * dx - inner(u, grad(v2)) * u2 * dx - eps * eps * inner(u, grad(v3)) * u3 * dx + \
inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx + eps * eps * inner(grad(u3),grad(v3)) * dx + \
- f1 * v1 * dx - f2 * v2 * dx - eps * eps * f3 * v3 * dx + \
- inner(theta, v) * ds(1)

# Compute solution
solve(F == 0, u, bcu,
      solver_parameters = {"newton_solver":{ "linear_solver" : "gmres"}})

file = File("auto_J_nonlinear_u_eps" + str(eps) + ".pvd")
file << u

#VD = FunctionSpace(mesh, "Lagrange", 1)
#du = project(div(u),VD)

#file2 = File("auto_J_nonlinear_div_u_eps" + str(eps) + ".pvd")
#file2 << du
