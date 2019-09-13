import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import datetime
from numpy.random import rand
#from ufl import div
from dolfin import *
import numpy

mesh = UnitSquareMesh(30, 30)
V = VectorElement("Lagrange", mesh.ufl_cell(), degree = 2, dim = 2)
P = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 1)
TH = V * P
VP = FunctionSpace(mesh, TH)

class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)
upperboundary = UpperBoundary()
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
upperboundary.mark(boundaries, 1)
ds = Measure('ds')[boundaries]

wind_shear_x = 10.0
theta = Constant((wind_shear_x, 0))

noslipbasin = DirichletBC(VP.sub(0), Constant((0, 0)), "on_boundary && x[1] < 1.0 - DOLFIN_EPS")
zerotop_u = DirichletBC(VP.sub(0).sub(1), 0, "on_boundary && x[1] > 1.0 - DOLFIN_EPS")

bcu = [noslipbasin, zerotop_u]

# Define variational problem
up = TrialFunction(VP)
u,p = split(up)
u1, u3 = split(u)
(v, q) = TestFunctions(VP)
v1, v3 = split(v)

up_ = Function(VP)
(u_, p_) = split(up_)
(u1_, u3_) = split(u_)

eps = 0.0000001

F = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx + eps*eps*inner(u, grad(u3)) * v3 * dx + eps*eps*inner(grad(u3),grad(v3)) * dx -p*div(v)*dx + q*div(u)*dx - inner(theta, v) * ds(1)

F = action(F, up_)
J  = derivative(F, up_, up)

problem = NonlinearVariationalProblem(F, up_, bcu, J)
solver  = NonlinearVariationalSolver(problem)
solver.solve()

(u,p) = up_.split(True)
print("Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2"))
print("Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2"))


(u,p) = up_.split(True)

ufile_pvd = File("velocity" + str(eps) + ".pvd")
ufile_pvd << u
pfile_pvd = File("pressure" + str(eps) + ".pvd")
pfile_pvd << p
