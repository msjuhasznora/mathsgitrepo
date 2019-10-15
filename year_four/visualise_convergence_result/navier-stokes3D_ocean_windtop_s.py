# this is a first simplified version.
# drop the Coriolis term
# stationary
# 2D ?

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import datetime

timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

parameters["std_out_all_processes"] = False;
meshsize = 20
mesh = UnitCubeMesh(meshsize, meshsize, meshsize)
values = []
# (order argument, optional argument: dim =, fill both in this case, unlike 2D )
V = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 3)
# Q = FunctionSpace(mesh, "Lagrange", 1)

u = Function(V)
# p = TrialFunction(Q)
v = TestFunction(V)
# q = TestFunction(Q)
u1, u2, u3 = split(u)
v1, v2, v3 = split(v)

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
# TODOthink about this, the bcp part. we originally didn't have this since we did not
# have a pressure part, but now we do
# bcp = []

wind_shear_x = 100.0
wind_shear_y = 100.0

# Define coefficients
f = Constant((0, 0, 0))
theta = Constant((wind_shear_x, wind_shear_y, 0))

eps = 1.0
while eps > DOLFIN_EPS:

    WF_stationary_anisotropic = dot(u, grad(u1)) * v1 * dx + dot(u, grad(u2)) * v2 * dx + \
        eps*eps*dot(u, grad(u3)) * v3 * dx + \
        inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx + eps*eps*inner(grad(u3),grad(v3)) * dx + \
        - inner(theta, v) * ds(1)

    a_wf_s_a = lhs(WF_stationary_anisotropic)
    L_wf_s_a = rhs(WF_stationary_anisotropic)

    # Assemble matrices
    A_wf_s_a = assemble(a_wf_s_a)

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True

    # Create files for storing solution
    ufile = File("results_ocean/resultsA" + str(timestamp) + "_mesh" + str(meshsize) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + "/velocity" + "_mesh" + str(meshsize) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + ".pvd")
    values.append(eps)

    b_wf_s_a = assemble(L_wf_s_a)
    [bc.apply(A_wf_s_a, b_wf_s_a) for bc in bcu]
    solve(A_wf_s_a, u, b_wf_s_a, "bicgstab", "default")

    # Save to file
    ufile << u

    norm_u1 = norm(u.sub(0), 'L2')
    values.append(norm_u1)

    eps = eps / 2.0
    np.savetxt("values_" + str(meshsize) + ".txt", values)
