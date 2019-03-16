# case: ocean with wind shear on top. Navier-Stokes + Chorin

# remaining steps:
# fix the weak convergence
# STEP: add new space C for the concentration with source term
# STEP: C function, (2,1,1) space, stability, BBL condition?

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import datetime

timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

parameters["std_out_all_processes"] = False;
meshsize = 20
mesh = UnitCubeMesh(meshsize, meshsize, meshsize)
values = []

# 1.) The idea here is to use both the 3-dimensional V, and the V_H & V_V function spaces,
# which are two and one dimensional, respectively.
# This is because at some point we update u_H, u_V separately, but at step3 of the Chorin method
# u = (u_H, u_V) is updated all at once.
# 2.) (order argument, optional argument: dim =, fill both in this case, unlike 2D )
V = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 3)
V_H = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 2)
V_V = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 1)
# V = V_H * V_V, see later if there is a shorter solution.
#it is needed for STEP3 of the Chorin, when we update u_h and u_v simultaneously.
Q = FunctionSpace(mesh, "Lagrange", 1)

u = TrialFunction(V)
u_H = TrialFunction(V_H)
u_V = TrialFunction(V_V)
p = TrialFunction(Q)

v = TestFunction(V)
v_H = TestFunction(V_H)
v_V = TestFunction(V_V)
q = TestFunction(Q)

u1, u2 = split(u_H)
v1, v2 = split(v_H)

# u_3d_1, u_3d_2, u_3d_3 = split(u)
v_3d_1, v_3d_2, v_3d_3 = split(v)

class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 1.0)
upperboundary = UpperBoundary()
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
upperboundary.mark(boundaries, 1)
ds = Measure('ds')[boundaries]

dt = 0.02
T = 1
alpha = 1.0
beta = 1.0

noslipbasin_3d = DirichletBC(V, (0, 0, 0), "on_boundary && x[2] < 1.0 - DOLFIN_EPS")
zerotop_3d = DirichletBC(V.sub(2), 0, "on_boundary && x[2] > 1.0 - DOLFIN_EPS")
noslipbasin_H = DirichletBC(V_H, (0, 0), "on_boundary && x[2] < 1.0 - DOLFIN_EPS")
noslipbasin_V = DirichletBC(V_V, 0, "on_boundary && x[2] < 1.0 - DOLFIN_EPS")
zerotop = DirichletBC(V_V, 0, "on_boundary && x[2] > 1.0 - DOLFIN_EPS")

bcu_3d = [noslipbasin_3d, zerotop_3d]
bcu_H = [noslipbasin_H]
bcu_V = [noslipbasin_V, zerotop]
# TODOthink about this, the bcp part. we originally didn't have this since we did not
# have a pressure part, but now we do
bcp = []

u_prev_H = Function(V_H)
u_prev_V = Function(V_V)
u_prev1, u_prev2 = split(u_prev_H)

u_prev_3d = Function(V)

u_next_H = Function(V_H)
u_next_V = Function(V_V)
u_next1, u_next2 = split(u_next_H)

u_next_3d = Function(V)

p_prev = Function(Q)
p_next = Function(Q)

wind_shear_x = 100.0
wind_shear_y = 100.0

# Define coefficients
k = Constant(dt)
f = Constant((0, 0, 0))
theta = Constant((wind_shear_x, wind_shear_y, 0))
theta1, theta2, theta3 = split(theta)

#Chorin's method. / Incremental pressure correction in 3 steps,
# as described in https://fenicsproject.org/pub/tutorial/pdf/fenics-tutorial-vol1.pdf
# on p57.

### in all 3 steps, the unknown function is
### denoted by u and p, whose type is by definition "TrialFunction"

# Define variational problem for step 1
### knowing u_prev, p_prev, we GET: u, ie u^*, the tentative velocity

### idea to fix the disappearing dependence of the first stop of u3. we could split the method to further steps,
### and we could introduce two different FEM spaces for u_h and u_3. in a first step we could calculate a tentative horizontal
### velocity, and then as a sort of correction, update u_3 in a scheme where u_3 is the only trial function and
### the equation for u_3 includes the results for u_H from the previous step.

eps = 1.0
while eps > DOLFIN_EPS:

    ### STEP1 / PART1 - update u_H - THUS we can !!!only!!! have V_H test functions
    # - inner(f, v) * dx + \
    F1_H = (1/k)*( (u1 - u_prev1)*v1 + (u2 - u_prev2)*v2 + \
        inner(u_prev_3d, grad(u_prev1)) * v1 * dx + inner(u_prev_3d, grad(u_prev2)) * v2 * dx + \
        - alpha * u_prev2 * v1 * dx + alpha * u_prev1 * v2 * dx + \
        inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx + \
        eps * beta * u_prev_V * v1 * dx + \
        p_prev.dx(0) * v1 * dx + p_prev.dx(1) * v2 * dx + \
        - (theta1 * v1 + theta2 * v2) * ds(1)

    a1 = lhs(F1_H)
    L1 = rhs(F1_H)

    ### STEP1 / PART2 - update u_V - THUS we can !!!only!!! have the V_V test function
    # - inner(f, v) * dx + \
    F1_V = (1/k)*( (u_next1 - u_prev1)*v_V + (u_next2 - u_prev2)*v_V + \
        ( u_next1 * u_next1.dx(0) + u_next2 * u_next1.dx(1) + u_V * u_next1.dx(2) ) * v_V * dx + \
        ( u_next1 * u_next2.dx(0) + u_next2 * u_next2.dx(1) + u_V * u_next2.dx(2) ) * v_V * dx + \
        - alpha * u_next2 * v_V * dx + alpha * u_next1 * v_V * dx + \
        inner(grad(u_next1),grad(v_V)) * dx + inner(grad(u_next2),grad(v_V)) * dx  + \
        eps * beta * u_V * v_V * dx + \
        p_prev.dx(0) * v_V * dx + p_prev.dx(1) * v_V * dx + \
        - (theta1 * v_V + theta2 * v_V) * ds(1)

    a2 = lhs(F1_V)
    L2 = rhs(F1_V)


    # Pressure update
    # Define variational problem for step 2
    ### this is where we GET p, ie p^*. from the previous
    ### step we solve for u^*, save it in u_next, and so at this
    ### point u_next contains u^* (see later at the time steps)
    ### STEP2 - update p
    a3 = inner(grad(p), grad(q))*dx
    L3 = - (1/k)*(u_next1.dx(0) + u_next2.dx(1) + u_next_V.dx(2)))*q*dx

    # Velocity update
    # Define variational problem for step 3
    ### we know u_next and p_next, here we GET u.
    ### STEP3 - update u, using the updated p.
    a4 = inner(u, v)*dx
    L4 = (u_next1 * v_3d_1 + u_next2 * v_3d_2 + u_next_V * v_3d_3)*dx - k*inner(p_next.dx(0) * v_3d_1 + p_next.dx(1) * v_3d_2 + p_next.dx(2) * v_3d_3)*dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)
    A4 = assemble(a4)

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True

    # Create files for storing solution
    ufile = File("results_ocean_H_V_idea/resultsA" + str(timestamp) + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + "/velocity" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + ".pvd")
    pfile = File("results_ocean_H_V_idea/resultsA" + str(timestamp) + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + "/pressure" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + ".pvd")
    values.append(eps)

    # Time-stepping
    t = dt
    while t < T + DOLFIN_EPS:

        # Compute tentative velocity step
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu_H]
        solve(A1, u_next_H.vector(), b1, "bicgstab", "default")

        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcu_V]
        solve(A2, u_next_V.vector(), b2, "bicgstab", "default")

        # Pressure correction
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcp]
        [bc.apply(p_next.vector()) for bc in bcp]
        solve(A3, p_next.vector(), b3, "bicgstab", prec)

        # Velocity correction
        b4 = assemble(L4)
        [bc.apply(A4, b4) for bc in bcu_3d]
        solve(A4, u_next_3d.vector(), b4, "bicgstab", "default")

        # Save to file
        ufile_3d << u_next_3d
        pfile << p_next

        u_update1, u_update2, u_update3 = split(u_next_3d)

        # Move to next time step
        u_prev1.assign(u_update1)
        u_prev2.assign(u_update2)
        u_prev_V.assign(u_update3)
        u_prev_3d.assign(u_next_3d)
        p_prev.assign(p_next)
        t += dt


#    norm_u = norm(u_next, 'L2')
#    norm_u1 = norm(u_next.sub(0), 'L2')
#    norm_u2 = norm(u_next.sub(1), 'L2')
#    norm_u3 = norm(u_next.sub(2), 'L2')
#    norm_u1_H1 = norm(u_next.sub(0), 'H1')
#    norm_u1_H10 = norm(u_next.sub(0), 'H10')
#    norm_u2_H1 = norm(u_next.sub(1), 'H1')
#    norm_u2_H10 = norm(u_next.sub(1), 'H10')
#    norm_u3_H1 = norm(u_next.sub(2), 'H1')
#    norm_u3_H10 = norm(u_next.sub(2), 'H10')
#    values.append(norm_u1)
#    values.append(norm_u2)
#    values.append(norm_u3)
#    values.append(norm_u3 / (norm_u1 + norm_u2))
#    values.append(norm_u1_H1)
#    values.append(norm_u1_H10)
#    values.append(norm_u2_H1)
#    values.append(norm_u2_H10)
#    values.append(norm_u3_H1)
#    values.append(norm_u3_H10)

    eps = eps / 2.0
    #np.savetxt("eps_norms_L2_H1_norm_values_u3_over_u1_u2_mesh_" + str(meshsize) + "_dt_" + str(dt) + ".txt", values)
