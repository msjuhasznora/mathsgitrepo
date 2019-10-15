# this version as it is does not work, because the formulation of the problem is nonlinear, while the function in it is of type TrialFunction. Nonlinear problems make sense with Function-type u.
# see https://fenicsproject.discourse.group/t/issue-with-gradient-inner-product/1402

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import datetime

print("Szia.")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Define coefficients
wind_shear_x = 50.0
wind_shear_y = 50.0
f = Constant((0, 0, 0))
theta = Constant((wind_shear_x, wind_shear_y, 0))
eps = 1.0

print(timestamp)

parameters["std_out_all_processes"] = False;
meshsize = 20
mesh = UnitCubeMesh(meshsize, meshsize, meshsize)
# (order argument, optional argument: dim =, fill both in this case, unlike 2D )
V = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 3)

u = TrialFunction(V)
v = TestFunction(V)

noslipbasin = DirichletBC(V, (0, 0, 0), "on_boundary && x[2] < 1.0 - DOLFIN_EPS")
zerotop = DirichletBC(V.sub(2), 0, "on_boundary && x[2] > 1.0 - DOLFIN_EPS")

bcu = [noslipbasin, zerotop]

F = inner(dot(u, grad(u)), v) * dx + inner(grad(u),grad(v)) * dx

# Create files for storing solution
ufile = File("out.pvd")

problem = NonlinearVariationalProblem(F, u, bcu)
solver = NonlinearVariationalSolver(problem)
solver.solve()

# Save to file
ufile << u
