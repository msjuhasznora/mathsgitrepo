# case: ocean with find shear on top. Navier-Stokes + Chorin

# remaining steps:
# fix the weak convergence
# STEP: add new space C for the concentration with source term
# STEP: C function, (2,1,1) space, stability, BBL condition?

import matplotlib.pyplot as plt
from dolfin import *

parameters["std_out_all_processes"] = False;
meshsize = 25
mesh = UnitCubeMesh(meshsize, meshsize, meshsize)

# (order argument, optional argument: dim =, fill both in this case, unlike 2D )
V = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 3)
Q = FunctionSpace(mesh, "Lagrange", 1)

u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)
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

dt = 0.01
T = 1
eps = 0.01
alpha = 1.0
beta = 1.0

noslipbasin = DirichletBC(V, (0, 0, 0), "on_boundary && x[2] < 1.0 - DOLFIN_EPS")
zerotop = DirichletBC(V.sub(2), 0, "on_boundary && x[2] > 1.0 - DOLFIN_EPS")

bcu = [noslipbasin, zerotop]
# TODOthink about this, the bcp part. we originally didn't have this since we did not
# have a pressure part, but now we do
bcp = []

u_prev = Function(V)
u_prev1, u_prev2, u_prev3 = split(u_prev)
u_next = Function(V)
u_next1, u_next2, u_next3 = split(u_next)
p_prev = Function(Q)
p_next = Function(Q)

wind_shear_x = 100.0
wind_shear_y = 100.0

# Define coefficients
k = Constant(dt)
f = Constant((0, 0, 0))
theta = Constant((wind_shear_x, wind_shear_y, 0))

#Chorin method. / Incremental pressure correction in 3 steps,
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

F1_anisotropic = (1/k)*( (u1 - u_prev1)*v1 + (u2 - u_prev2)*v2 + eps*eps*(u3 - u_prev3)*v3 ) * dx + \
     inner(u_prev, grad(u_prev1)) * v1 * dx + inner(u_prev, grad(u_prev2)) * v2 * dx + \
     eps*eps*inner(u_prev, grad(u_prev3)) * v3 * dx + \
     - alpha * u_prev2 * v1 * dx + alpha * u_prev1 * v2 * dx + \
     inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx + eps*eps*inner(grad(u3),grad(v3)) * dx + \
     eps * beta * u_prev3 * v1 * dx - eps * beta * u_prev1 * v3 * dx + \
     - inner(f, v) * dx + \
     - inner(theta, v) * ds(1)

# here we only have u1 and u2, the third component is gone, the method does not converge. to be fixed.
F1_hydrostatic = (1/k)*( (u1 - u_prev1)*v1 + (u2 - u_prev2)*v2 ) * dx + \
     inner(u_prev, grad(u_prev1)) * v1 * dx + inner(u_prev, grad(u_prev2)) * v2 * dx + \
     - alpha * u_prev2 * v1 * dx + alpha * u_prev1 * v2 * dx + \
     inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx + \
     - inner(f, v) * dx + \
     - inner(theta, v) * ds(1)

a1 = lhs(F1_anisotropic)
L1 = rhs(F1_anisotropic)

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
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Create files for storing solution
ufile = File("resultsA" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + "/velocity.pvd")
pfile = File("resultsA" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "_eps" + str(eps) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + "/pressure.pvd")

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:

    # Compute tentative velocity step
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u_next.vector(), b1, "bicgstab", "default")

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

    # Move to next time step
    u_prev.assign(u_next)
    p_prev.assign(p_next)
    t += dt
