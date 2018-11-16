# case: ocean, with wind on the surface

# this is made for the case of u3 = 0,
# inspired by the expectation and also the results of the 3D version resulting in a close-to-zero vertical velocity overall.

import matplotlib.pyplot as plt
from dolfin import *

parameters["std_out_all_processes"] = False;
meshsize = 20
mesh = UnitCubeMesh(meshsize, meshsize, meshsize)

# order, dimension
V = VectorFunctionSpace(mesh, "CG", 1, dim = 2)
Q = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)
u1, u2 = split(u)
v1, v2 = split(v)

class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 1.0)
upperboundary = UpperBoundary()
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
upperboundary.mark(boundaries, 1)
ds = Measure('ds')[boundaries]

dt = 0.005
T = 1
alpha = 1.0
beta = 1.0

noslipbasin = DirichletBC(V, (0, 0), "on_boundary && x[2] < 1.0 - DOLFIN_EPS")

bcu = [noslipbasin]
# TODOthink about this, the bcp part. we originally didn't have this since we did not
# have a pressure part, but now we do
bcp = []

u_prev = Function(V)
u_prev1, u_prev2 = split(u_prev)
u_next = Function(V)
u_next1, u_next2 = split(u_next)
p_prev = Function(Q)
p_next = Function(Q)

wind_shear_x = 100.0
wind_shear_y = 100.0

# Define coefficients
k = Constant(dt)
f = Constant((0, 0))
theta = Constant((wind_shear_x, wind_shear_y))

#Chorin.

### in all 3 steps, the unknown function is
### denoted by u and p, whose type is by definition "TrialFunction"

# Define variational problem for step 1
### knowing u_prev, p_prev, we GET: u, ie u^*, the tentative velocity

F1_hydrostatic = (1/k)*( (u1 - u_prev1)*v1 + (u2 - u_prev2)*v2 ) * dx + \
     (u_prev1 * u_prev1.dx(0) + u_prev2 * u_prev1.dx(1)) * v1 * dx + (u_prev1 * u_prev2.dx(0) + u_prev2 * u_prev2.dx(1)) * v2 * dx + \
     - alpha * u_prev2 * v1 * dx + alpha * u_prev1 * v2 * dx + \
     inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx + \
     - inner(f, v) * dx + \
     - inner(theta, v) * ds(1)

a1 = lhs(F1_hydrostatic)
L1 = rhs(F1_hydrostatic)

# Pressure update
# Define variational problem for step 2
### this is where we GET p, ie p^*. from the previous
### step we solve for u^*, save it in u_next, and so at this
### point u_next contains u^* (see later at the time steps)
a2 = (p.dx(0)*q.dx(0) + p.dx(1)*q.dx(1) + p.dx(2)*q.dx(2))*dx
L2 = - (1/k)*(u_next1.dx(0)+ u_next2.dx(1))*q*dx

# Velocity update
# Define variational problem for step 3
### we know u_next and p_next, here we GET u.
a3 = inner(u, v)*dx
L3 = inner(u_next, v)*dx - k*(p_next.dx(0)*v1 + p_next.dx(1)*v2)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Create files for storing solution
ufile = File("resultsH" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + "/velocity" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + ".pvd")
pfile = File("resultsH" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + "/pressure" + "_mesh" + str(meshsize) + "_dt" + str(dt) + "ws" + str(wind_shear_x) + "_" + str(wind_shear_y) + ".pvd")

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
