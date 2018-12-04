# case: ocean with find shear on top. Navier-Stokes + Chorin

# remaining steps:
# fix the weak convergence
# STEP: add new space C for the concentration with source term
# STEP: C function, (2,1,1) space, stability, BBL condition?

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np

parameters["std_out_all_processes"] = False;
meshsize = 15
mesh = UnitCubeMesh(meshsize, meshsize, meshsize)
values = []
# (order argument, optional argument: dim =, fill both in this case, unlike 2D )
U = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 3)
U_H = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 2)
U_V = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 1)
Q = FunctionSpace(mesh, "Lagrange", 1)
C = FunctionSpace(mesh, "Lagrange", 1)

u = TrialFunction(U)
u_h = TrialFunction(U_H)
u_v = TrialFunction(U_V)
p = TrialFunction(Q)
c = TrialFunction(C)
v = TestFunction(U)
v_h = TrialFunction(U_H)
v_v = TrialFunction(U_V)
q = TestFunction(Q)
d = TestFunction(C)
u1, u2, u3 = split(u)
v1, v2, v3 = split(v)
u_h1, u_h2 = split(u_h)
v_h1, v_h2 = split(v_h)

class LowerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 0.0)
lowerboundary = LowerBoundary()
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
lowerboundary.mark(boundaries, 1)
ds = Measure('ds')[boundaries]

dt = 0.01
T = 1
alpha = 1.0
beta = 1.0

zerolateralaboveU = DirichletBC(V, (0, 0, 0), "on_boundary && x[0] > 0.0 + DOLFIN_EPS")
zerolateralaboveC = DirichletBC(C, 0, "on_boundary && x[2] > 0.0 + DOLFIN_EPS")

bcu = [zerolateralaboveU]
bcp = []
bcc = [zerolateralaboveC]

c_next = Function(C)
c_prev = Function(C)
p_next = Function(Q)
p_prev = Function(Q)
u_next = Function(U)
u_prev = Function(U)
u_prev1, u_prev2, u_prev3 = split(u_prev)
u_h_next = Function(U_H)
u_v_next = Function(U_H)

wind_shear_x = 100.0
wind_shear_y = 100.0

delta = PointSource(C, Point(0.5, 0.5, 0.5), 100)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0, 0))
theta = Constant((wind_shear_x, wind_shear_y, 0))

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

    F0 = (c - c_prev) * d * dx - div(u_prev) * c * d * dx - inner(u_prev, grad(d)) * c * dx + \
            eps * c.dx(0) * d.dx(0) * dx - eps * c.dx(0) * d * ds(1) + c.dx(1) * d.dx(1) * dx + c.dx(2) * d.dx(2) * dx
    a0 = lhs(F0)
    L0 = rhs(F0)

    F1 = (1/k)*( (u1 - u_prev1)*v1 + (u2 - u_prev2)*v2 + \
        inner(u_prev, grad(u_prev1)) * v1 * dx + inner(u_prev, grad(u_prev2)) * v2 * dx + \
        - alpha * u_prev2 * v1 * dx + alpha * u_prev1 * v2 * dx + \
        inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx  + \
        eps * beta * u_prev3 * v1 * dx \
        - inner(f, v) * dx + \
        - inner(theta, v) * ds(1)

    a1 = lhs(F1)
    L1 = rhs(F1)

    eps*eps*(u3 - u_prev3)*v3 + eps*eps*inner(u_prev, grad(u_prev3)) * v3 * dx
    + eps*eps*inner(grad(u3),grad(v3)) * dx - eps * beta * u_prev1 * v3 * dx +

    # Pressure update
    # Define variational problem for step 2
    ### this is where we GET p, ie p^*. from the previous
    ### step we solve for u^*, save it in u_next, and so at this
    ### point u_next contains u^* (see later at the time steps)
    a2 = inner(grad(p), grad(q))*dx
    L2 = - (1/k)*div(u_next)*q*dx

    # Velocity update
    # Define variational problem for step 3
    ### we know u_next and p_next, here we GET u.
    a3 = inner(u, v)*dx
    L3 = inner(u_next, v)*dx - k*inner(grad(p_next), v)*dx

    # Assemble matrices
    A0 = assemble(a0)
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True

    # Create files for storing solution
    ufile = File("resultsC" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + "/velocity" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + ".pvd")
    pfile = File("resultsC" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + "/pressure" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + ".pvd")
    cfile = File("resultsC" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + "/concentration" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + ".pvd")

    values.append(eps)

    # Time-stepping
    t = dt
    while t < T + DOLFIN_EPS:

        b0 = assemble(L0)
        delta.apply(b0)
        [bc.apply(A0, b0) for bc in bcc]
        solve(A0, c_next.vector(), b0, "bicgstab", prec)

        # Compute tentative velocity step
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, u_h_next.vector(), b1, "bicgstab", "default")

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
        ufile << u_next
        pfile << p_next
        cfile << c_next

        # Move to next time step
        u_prev.assign(u_next)
        p_prev.assign(p_next)
        c_prev.assign(c_next)
        t += dt


    norm_u = norm(u_next)
    norm_u3 = norm(u_next.sub(2))

    values.append(norm_u3 / norm_u)

    eps = eps / 2.0
    np.savetxt("eps_norm_values_mesh" + str(meshsize) + "_dt_" + str(dt) + ".txt", values)
