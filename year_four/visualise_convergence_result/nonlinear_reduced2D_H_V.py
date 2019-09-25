# The anisotropic model works nicely with the (2,2) degree scenario, the hydrostatic scheme works for (2,1).
# The former does not work for (2,1) as it develops strange unnatural layers in the pressure.
# The latter does not work for (2,2) as the Newton iterations do not converge --- probably because we do not have the first derivative of u3 in the scheme and u3 is second order in that case.

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import datetime
from numpy.random import rand
#from ufl import div
from dolfin import *
import numpy

degree_vertical_anis = 2
degree_vertical_hydr = 1
epsilon_lower_limit = 0.00001 #1.0e-07

resultsfolder = "results_2D_H_V_degree_anis" + str(degree_vertical_anis) + "_degree_hydr" + str(degree_vertical_hydr) + "/"

mesh = UnitSquareMesh(30, 30)
V_hor = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 2)
V_vert_hydr = FiniteElement("Lagrange", mesh.ufl_cell(), degree = degree_vertical_hydr)
V_vert_anis = FiniteElement("Lagrange", mesh.ufl_cell(), degree = degree_vertical_anis)
V_hydr = V_hor * V_vert_hydr
V_anis = V_hor * V_vert_anis
P = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 1)
VP_hydr_element = V_hydr * P # linear in the vertical velocity space
VP_anis_element = V_anis * P # Taylor - Hood
VP_hydr = FunctionSpace(mesh, VP_hydr_element)
VP_anis = FunctionSpace(mesh, VP_anis_element)

class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)
upperboundary = UpperBoundary()
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
upperboundary.mark(boundaries, 1)
ds = Measure('ds')[boundaries]

wind_shear_x = 10.0
theta = Constant((wind_shear_x, 0.0))

noslipbasin_hydr = DirichletBC(VP_hydr.sub(0), Constant((0, 0)), "on_boundary && x[1] < 1.0 - DOLFIN_EPS")
noslipbasin_anis = DirichletBC(VP_anis.sub(0), Constant((0, 0)), "on_boundary && x[1] < 1.0 - DOLFIN_EPS")
zerotopvertical_hydr = DirichletBC(VP_hydr.sub(0).sub(1), 0, "on_boundary && x[1] > 1.0 - DOLFIN_EPS")
zerotopvertical_anis = DirichletBC(VP_anis.sub(0).sub(1), 0, "on_boundary && x[1] > 1.0 - DOLFIN_EPS")

bcu_hydr = [noslipbasin_hydr, zerotopvertical_hydr]
bcu_anis = [noslipbasin_anis, zerotopvertical_anis]

# **********************************************
# *** Define hydrostatic variational problem ***
# **********************************************

up = TrialFunction(VP_hydr)
u,p = split(up)
u1, u3 = split(u)
(v, q) = TestFunctions(VP_hydr)
v1, v3 = split(v)

up_ = Function(VP_hydr)
(u_, p_) = split(up_)
(u1_, u3_) = split(u_)

# the hydrostatic weak formulation is constructed with the vertical velocity space being of degree 1 and the additional constraint p.dx(1) * q.dx(1) * dx representing that we have a hydrostatic pressure. using a lower degree for the vertical velocities for the case of the primitive equations come from the article of Danilov, Gennady, Schroter, 2002 (even though they use elementwise constant representations)
F_hydr = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx - p * div(v) * dx + q * div(u) * dx + p.dx(1) * q.dx(1) * dx - inner(theta, v) * ds(1)

F_hydr = action(F_hydr, up_)
J_hydr  = derivative(F_hydr, up_, up)

problem_hydr = NonlinearVariationalProblem(F_hydr, up_, bcu_hydr, J_hydr)
solver  = NonlinearVariationalSolver(problem_hydr)
solver.solve()

up_sol_hydr = Function(VP_hydr)
(u_sol_hydr, p_sol_hydr) = split(up_sol_hydr)
(u1_sol_hydr, u3_sol_hydr) = split(u_sol_hydr)

(u_sol_hydr, p_sol_hydr) = up_.split(True)
(u1_sol_hydr, u3_sol_hydr) = u_sol_hydr.split(True)

print("Hydrostatic. Norm of velocity coefficient vector: %.15g" % u_sol_hydr.vector().norm("l2"))
print("Hydrostatic. Norm of horizontal velocity coefficient vector: %.15g" % u1_sol_hydr.vector().norm("l2"))
print("Hydrostatic. Norm of vertical velocity coefficient vector: %.15g" % u3_sol_hydr.vector().norm("l2"))
print("Hydrostatic. Norm of pressure coefficient vector: %.15g" % p_sol_hydr.vector().norm("l2"))

(u,p) = up_.split(True)

ufile_pvd_hydr = File(resultsfolder + "velocity_hydr.pvd")
ufile_pvd_hydr << u
pfile_pvd_hydr = File(resultsfolder + "pressure_hydr.pvd")
pfile_pvd_hydr << p

# **********************************************
# *** Define anisotropic variational problem ***
# **********************************************

eps = 1.0

while eps > epsilon_lower_limit:
    
    up = TrialFunction(VP_anis)
    u,p = split(up)
    u1, u3 = split(u)
    (v, q) = TestFunctions(VP_anis)
    v1, v3 = split(v)

    up_ = Function(VP_anis)
    (u_, p_) = split(up_)
    (u1_, u3_) = split(u_)

    # the anisotropic weak formulation is created using the Taylor-Hood elements, the vertical velocity is from a quadratic space. Using a degree 1 vertical velocity space in the anisotropic case we have a strange layered unnatural pressure.
    F_anis = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx + eps*eps*inner(u, grad(u3)) * v3 * dx + eps*eps*inner(grad(u3),grad(v3)) * dx - p * div(v) * dx + q * div(u) * dx - inner(theta, v) * ds(1)

    F_anis = action(F_anis, up_)
    J_anis  = derivative(F_anis, up_, up)

    problem_anis = NonlinearVariationalProblem(F_anis, up_, bcu_anis, J_anis)
    solver  = NonlinearVariationalSolver(problem_anis)
    solver.solve()

    (u,p) = up_.split(True)
    (u1, u3) = u.split(True)

    up_project_hydr = Function(VP_hydr)
    up_project_hydr = project(up_,VP_hydr)
    (u_project_hydr, p_project_hydr) = up_project_hydr.split(True)
    (u1_project_hydr, u3_project_hydr) = u_project_hydr.split(True)
    
    up_interpolate_hydr = Function(VP_hydr)
    up_interpolate_hydr = interpolate(up_,VP_hydr)
    (u_interpolate_hydr, p_interpolate_hydr) = up_interpolate_hydr.split(True)
    (u1_interpolate_hydr, u3_interpolate_hydr) = u_interpolate_hydr.split(True)
    
    print("Epsilon: " + str(eps))
    
    print("Hydrostatic. u: %.15g" % u_sol_hydr.vector().norm("l2"))
    print("Hydrostatic. u1: %.15g" % u1_sol_hydr.vector().norm("l2"))
    print("Hydrostatic. u3: %.15g" % u3_sol_hydr.vector().norm("l2"))
    print("Hydrostatic. p: %.15g" % p_sol_hydr.vector().norm("l2"))

    print("Anistropic Projected. u: %.15g" % u_project_hydr.vector().norm("l2"))
    print("Anistropic Projected. u1: %.15g" % u1_project_hydr.vector().norm("l2"))
    print("Anistropic Projected. u3: %.15g" % u3_project_hydr.vector().norm("l2"))
    print("Anistropic Projected. p: %.15g" % p_project_hydr.vector().norm("l2"))
    
    print("Anistropic Interpolated. u: %.15g" % u_interpolate_hydr.vector().norm("l2"))
    print("Anistropic Interpolated. u1: %.15g" % u1_interpolate_hydr.vector().norm("l2"))
    print("Anistropic Interpolated. u3: %.15g" % u3_interpolate_hydr.vector().norm("l2"))
    print("Anistropic Interpolated. p: %.15g" % p_interpolate_hydr.vector().norm("l2"))

    print("Anistropic. u: %.15g" % u.vector().norm("l2"))
    print("Anistropic. u1: %.15g" % u1.vector().norm("l2"))
    print("Anistropic. u3: %.15g" % u3.vector().norm("l2"))
    print("Anistropic. p: %.15g" % p.vector().norm("l2"))
    
    print("Anistropic Interpolated - Hydrostatic. u: %.15g" % (u_interpolate_hydr.vector() - u_sol_hydr.vector()).norm("l2"))
    print("Anistropic Interpolated - Hydrostatic. u1: %.15g" % (u1_interpolate_hydr.vector() - u1_sol_hydr.vector()).norm("l2"))
    print("Anistropic Interpolated - Hydrostatic. u3: %.15g" % (u3_interpolate_hydr.vector() - u3_sol_hydr.vector()).norm("l2"))
    print("Anistropic Interpolated - Hydrostatic. p: %.15g" % (p_interpolate_hydr.vector() - p_sol_hydr.vector()).norm("l2"))
    
    print("Anistropic Projected - Hydrostatic. u: %.15g" % (u_project_hydr.vector() - u_sol_hydr.vector()).norm("l2"))
    print("Anistropic Projected - Hydrostatic. u1: %.15g" % (u1_project_hydr.vector() - u1_sol_hydr.vector()).norm("l2"))
    print("Anistropic Projected - Hydrostatic. u3: %.15g" % (u3_project_hydr.vector() - u3_sol_hydr.vector()).norm("l2"))
    print("Anistropic Projected - Hydrostatic. p: %.15g" % (p_project_hydr.vector() - p_sol_hydr.vector()).norm("l2"))

    (u,p) = up_.split(True)

    ufile_pvd_anis = File(resultsfolder + "velocity_anis" + str(eps) + ".pvd")
    pfile_pvd_anis = File(resultsfolder + "pressure_anis" + str(eps) + ".pvd")
    ufile_pvd_anis << u
    pfile_pvd_anis << p

    eps = eps / 2.0

# *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- #

# improvement ideas:

# 0) add C

# i) the scheme depends epsilon-freely on the complete u. why do the Newton iterations not converge for the case when we have degree=2 for the vertical velocity? is it that the scheme contains only u3, but not grad(u3), and it is a 2-degree space?

# ii) consider making it time dependent

# iii) 3D in space

# iv) periodicity in the x direction would make the domain into a tube with upper wind traction and fully x-directional circulation. Having a 0 y-directional velocity would be ok in itself, but with that there is not much to visualise as the scheme then does not depend on epsilon. so I think it is better to have the classical domain in order to make a point with the visualisation.

# v) make the output nicer, append into one file for better visibility

# vi) the dimension and determinant of the stiffness matrix:

#a = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx + eps*eps*inner(u, grad(u3)) * v3 * dx + eps*eps*inner(grad(u3),grad(v3)) * dx
#A = assemble(a)
#J_mat = assemble(a)
#J_array = J_mat.array()
#detJ = numpy.linalg.det(J_array)
#print(detJ)
#np.savetxt("stiffnessmatrix.txt", A.array())
