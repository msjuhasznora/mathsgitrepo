# nonlinear solver
# gmres solver is used, mumps caused the process to be "killed"
# the weak form of the divergence-free condition is missing.
# the Jacobian is computed in the code

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

wind_shear_x = 10.0
wind_shear_y = 10.0
theta = Constant((wind_shear_x, wind_shear_y, 0))

# Choice of nonlinear coefficient

c1 = 1.0
c2 = 1.0
c3 = 1.0

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
u1, u2, u3 = split(u)
v1, v2, v3 = split(v)
u_ = Function(V)
F = c1 * inner(u, grad(u1)) * v1 * dx + c2 * inner(u, grad(u2)) * v2 * dx + \
inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx + \
+ c3 * inner(u, grad(u3)) * v3 * dx + \
+ c3 * inner(grad(u3),grad(v3)) * dx + \
- inner(theta, v) * ds(1)

F  = action(F, u_)
J  = derivative(F, u_, u)

# Compute solution
problem = NonlinearVariationalProblem(F, u_, bcu, J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['relaxation_parameter'] = 1.0
prm['newton_solver']['linear_solver'] = 'gmres'
prm['newton_solver']['preconditioner'] = 'ilu'

solver.solve()

file = File("nonlinear_u" + str(c3) + ".pvd")
file << u_
