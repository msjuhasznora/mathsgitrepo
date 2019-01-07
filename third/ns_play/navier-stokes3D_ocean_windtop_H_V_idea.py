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
# (order argument, optional argument: dim =, fill both in this case, unlike 2D )
# V = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 3)
V_H = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 2)
V_V = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 1)
# for this something like V = V_H * V_V would be needed, which hasn't work so far.
#it is needed for STEP3 of the Chorin, when we update u_h and u_v simultaneously.
Q = FunctionSpace(mesh, "Lagrange", 1)

# u = TrialFunction(V)
u_H = TrialFunction(V_H)
u_V = TrialFunction(V_V)
p = TrialFunction(Q)

# v = TestFunction(V)
v_H = TestFunction(V_H)
v_V = TestFunction(V_V)
q = TestFunction(Q)

u1, u2 = split(u_H)
v1, v2 = split(v_H)

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

# noslipbasin = DirichletBC(V, (0, 0, 0), "on_boundary && x[2] < 1.0 - DOLFIN_EPS")
noslipbasin_H = DirichletBC(V_H, (0, 0), "on_boundary && x[2] < 1.0 - DOLFIN_EPS")
noslipbasin_V = DirichletBC(V_V, 0, "on_boundary && x[2] < 1.0 - DOLFIN_EPS")
zerotop = DirichletBC(V_V, 0, "on_boundary && x[2] > 1.0 - DOLFIN_EPS")

bcu_H = [noslipbasin_H]
bcu_V = [noslipbasin_V, zerotop]
# TODOthink about this, the bcp part. we originally didn't have this since we did not
# have a pressure part, but now we do
bcp = []

u_prev_H = Function(V_H)
u_prev_V = Function(V_V)
u_prev1, u_prev2 = split(u_prev_H)

u_next_H = Function(V_H)
u_next_V = Function(V_V)
u_next1, u_next2 = split(u_next_H)

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

    F1_H = (1/k)*( (u1 - u_prev1)*v1 + (u2 - u_prev2)*v2 + \
        inner(u_prev, grad(u_prev1)) * v1 * dx + inner(u_prev, grad(u_prev2)) * v2 * dx + \
        - alpha * u_prev2 * v1 * dx + alpha * u_prev1 * v2 * dx + \
        inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx  + \
        eps * beta * u_prev3 * v1 * dx + \
        p_prev.dx(0) * v1 * dx + p_prev.dx(1) * v2 * dx + \
        # - inner(f, v) * dx + \
        - (theta1 * v1 + theta2 * v2 + theta3 * v_V) * ds(1)

    a1_H = lhs(F1_H)
    L1_H = rhs(F1_H)

    F1_V = (1/k)*( (u_next1 - u_prev1)*v1 + (u_next2 - u_prev2)*v2 + \
        ( u_next1 * u_next1.dx(0) + u_next2 * u_next1.dx(1) + u_V * u_next1.dx(2) ) * v1 * dx + \
        ( u_next1 * u_next2.dx(0) + u_next2 * u_next2.dx(1) + u_V * u_next2.dx(2) ) * v2 * dx + \
        - alpha * u_next2 * v1 * dx + alpha * u_next1 * v2 * dx + \
        inner(grad(u_next1),grad(v1)) * dx + inner(grad(u_next2),grad(v2)) * dx  + \
        eps * beta * u_V * v1 * dx + \
        p_prev.dx(0) * v1 * dx + p_prev.dx(1) * v2 * dx + \
        # - inner(f, v) * dx + \
        - (theta1 * v1 + theta2 * v2 + theta3 * v_V) * ds(1)

    a1_V = lhs(F1_V)
    L1_V = rhs(F1_V)


    # Pressure update
    # Define variational problem for step 2
    ### this is where we GET p, ie p^*. from the previous
    ### step we solve for u^*, save it in u_next, and so at this
    ### point u_next contains u^* (see later at the time steps)
    a2 = inner(grad(p), grad(q))*dx
    L2 = - (1/k)*(u_next1.dx(0) + u_next2.dx(1) + u_next_V.dx(2)))*q*dx

    # Velocity update
    # Define variational problem for step 3
    ### we know u_next and p_next, here we GET u.
    a3 = inner(u1 * v1 + u2 * v2 + u_V * v_V)*dx
    L3 = (u_next1 * v1 + u_next2 * v2 + u_next_V * v_V)*dx - k*inner(p_next.dx(0) * v1 + p_next.dx(1) * v2 + p_next.dx(2) * v_V)*dx

    # Assemble matrices
    A1_H = assemble(a1_H)
    A1_V = assemble(a1_V)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True

    # Create files for storing solution
    ufile = File("results_ocean_H_V/resultsA" + str(timestamp) + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + "/velocity" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + ".pvd")
    pfile = File("results_ocean_H_V/resultsA" + str(timestamp) + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + "/pressure" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + ".pvd")
    values.append(eps)

    # Time-stepping
    t = dt
    while t < T + DOLFIN_EPS:

        # Compute tentative velocity step
        b1_H = assemble(L1_H)
        [bc.apply(A1_H, b1_H) for bc in bcu_H]
        solve(A1_H, u_next_H.vector(), b1_H, "bicgstab", "default")

        b1_V = assemble(L1_V)
        [bc.apply(A1_V, b1_V) for bc in bcu_V]
        solve(A1_V, u_next_V.vector(), b1_V, "bicgstab", "default")

        # Pressure correction
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcp]
        [bc.apply(p_next.vector()) for bc in bcp]
        solve(A2, p_next.vector(), b2, "bicgstab", prec)

        # Velocity correction
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcu]
        solve(A3, u_next.vector(), b3, "bicgstab", "default")

        # Save to file
        ufile_H << u_next_H
        ufile_V << u_next_V
        pfile << p_next

        # Move to next time step
        u_prev_H.assign(u_next_H)
        u_prev_V.assign(u_next_V)
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
