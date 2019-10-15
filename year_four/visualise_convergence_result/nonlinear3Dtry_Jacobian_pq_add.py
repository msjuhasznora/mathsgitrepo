import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import datetime
from numpy.random import rand
#from ufl import div
from dolfin import *
import numpy

# based on https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/documented/stokes-taylor-hood/demo_stokes-taylor-hood.py

# ideas how to move on: https://fenicsproject.discourse.group/t/how-to-create-a-divergence-free-vectorfunctionspace/1417/9

# gmres nonlinear solver
# add the weak divergence-free condition
# maybe add a velocity forcing term
# for now it ends with UMFPACK V5.7.1 (Oct 10, 2014): ERROR: out of memory

# here we add the weak form of the divergence-free condition into the full weak form and solve
# for (u,p)
# the divergence-free condition is represented like this also in the Stokes demo.

#V = VectorFunctionSpace(mesh, "CG", 2)
#P = FunctionSpace(mesh, "CG", 1)
#deprecated: VP = MixedFunctionSpace([V, P])
# deprecated: VP = V * P

# Create mesh and define function space
mesh = UnitCubeMesh(20, 20, 20)
V = VectorElement("Lagrange", mesh.ufl_cell(), degree = 2, dim = 3)
P = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 1)
TH = V * P
VP = FunctionSpace(mesh, TH)

# Define boundary conditions
class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 1.0)
upperboundary = UpperBoundary()
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
upperboundary.mark(boundaries, 1)
ds = Measure('ds')[boundaries]

wind_shear_x = 10.0
wind_shear_y = 10.0
theta = Constant((wind_shear_x, wind_shear_y, 0))

noslipbasin = DirichletBC(VP.sub(0), (0, 0, 0), "on_boundary && x[2] < 1.0 - DOLFIN_EPS")
zerotop_u = DirichletBC(VP.sub(0).sub(2), 0, "on_boundary && x[2] > 1.0 - DOLFIN_EPS")

bcu = [noslipbasin, zerotop_u]

# Define variational problem
up = TrialFunction(VP)
u,p = split(up)
u1, u2, u3 = split(u)
(v, q) = TestFunctions(VP)
v1, v2, v3 = split(v)

up_ = Function(VP)
(u_, p_) = split(up_)
(u1_, u2_, u3_) = split(u_)

F = inner(u, grad(u1)) * v1 * dx + inner(u, grad(u2)) * v2 * dx + inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx + inner(u, grad(u3)) * v3 * dx + inner(grad(u3),grad(v3)) * dx - inner(p, div(v))*dx + inner(q, div(u))*dx - inner(theta, v) * ds(1)

F = action(F, up_)
J  = derivative(F, up_, up)

problem = NonlinearVariationalProblem(F, up_, bcu, J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
#prm['newton_solver']['linear_solver'] = 'gmres'
#prm['newton_solver']['preconditioner'] = 'ilu'

solver.solve()

#solve(F == 0, up_, bcu, J, solver_parameters = {"newton_solver":{ "linear_solver" : "gmres"}})

(u,p) = up_.split(True)
print("Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2"))
print("Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2"))

(u,p) = up_.split(True)

ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p
