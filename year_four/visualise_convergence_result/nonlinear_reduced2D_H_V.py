import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import datetime
from numpy.random import rand
#from ufl import div
from dolfin import *
import numpy

mesh = UnitSquareMesh(30, 30)
V_hor = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 2)
V_vert_hydr = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 1)
V_vert_anis = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 2)
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

ufile_pvd = File("velocity_hydr.pvd")
ufile_pvd << u
pfile_pvd = File("pressure_hydr.pvd")
pfile_pvd << p

# **********************************************
# *** Define anisotropic variational problem ***
# **********************************************

up = TrialFunction(VP_anis)
u,p = split(up)
u1, u3 = split(u)
(v, q) = TestFunctions(VP_anis)
v1, v3 = split(v)

up_ = Function(VP_anis)
(u_, p_) = split(up_)
(u1_, u3_) = split(u_)

eps = 1.0
# create a while cycle that uses eps = eps /2 until the norms of two consequtive solutions are really close.

# the anisotropic weak formulation is created using the Taylor-Hood elements, the vertical velocity is from a quadratic space. Using a degree 1 vertical velocity space in the anisotropic case we have a strange layered unnatural pressure.
F_anis = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx + eps*eps*inner(u, grad(u3)) * v3 * dx + eps*eps*inner(grad(u3),grad(v3)) * dx - p * div(v) * dx + q * div(u) * dx - inner(theta, v) * ds(1)

F_anis = action(F_anis, up_)
J_anis  = derivative(F_anis, up_, up)

problem_anis = NonlinearVariationalProblem(F_anis, up_, bcu_anis, J_anis)
solver  = NonlinearVariationalSolver(problem_anis)
#prm = solver.parameters
#prm['newton_solver']['absolute_tolerance'] = 1E-8
#prm['newton_solver']['relative_tolerance'] = 1E-6
#prm['newton_solver']['maximum_iterations'] = 100
solver.solve()

(u,p) = up_.split(True)
(u1, u3) = u.split(True)

# project u3 to the 1-degree vertical velocity space?

print("Anistropic. Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2"))
print("Anistropic. Norm of horizontal velocity coefficient vector: %.15g" % u1.vector().norm("l2"))
print("Anistropic. Norm of vertical velocity coefficient vector: %.15g" % u3.vector().norm("l2"))
print("Anistropic. Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2"))

(u,p) = up_.split(True)

ufile_pvd = File("velocity_anis" + str(eps) + ".pvd")
ufile_pvd << u
pfile_pvd = File("pressure_anis" + str(eps) + ".pvd")
pfile_pvd << p


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
