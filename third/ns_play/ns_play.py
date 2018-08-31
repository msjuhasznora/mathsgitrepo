# what we had:
# time independent / stationary Stokes -- Taylor - Hood
# some details & importance of TH
# time dependent 2D N-S.
# only Dirichlet BCs
# find out how it works and why, then start modifying.
# time option: Chorin (no pressure at step1/3)
# time option: IPCS, prev pressure at step 1/3 - incremented accuracy.
# figure out how to implement NBCs

# steps:

# DONE STEP 1: modify the 2D L-shape domain to a 3D domain.
# DONE STEP 2: specify the order and the dimension for the velocity FEM space.
# (order argument, optional argument: dim =, fill both in this case, unlike 2D )
# DONE STEP3: change from Chorin to IPCS (in all 3 steps we have p0,
# which stands for the previous time step.)
# DONE STEP4: use meaningful notations instead of "u_" etc.
# DONE STEP5: redefine weak form with the actual one we use, with epsilon etc
# DONE STEP: mark the Neumann boundary

# STEP: Neumann BCs.

# STEP: check weak convergence for the ocean version, and air w/o C.

# STEP: add new space C.
# STEP: add source term
# STEP: C function, (2,1,1) space, stability, BBL condition?

# later:
# compare original Chorin & IPCS


import matplotlib.pyplot as plt
from dolfin import *

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Load mesh from file
mesh = UnitCubeMesh(5,5,5)

plt.figure()
plot(mesh)

#mesh = Mesh("meshes/unitsquare_32_32.xml.gz")

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2, dim = 3)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)
u1, u2, u3 = split(u)
v1, v2, v3 = split(v)

class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 1.0)
#upperboundary = UpperBoundary()
#boundaries = FacetFunction('size_t', mesh)
#boundaries = FacetFunction('size_t', mesh)​
#boundaries.set_all(0)
#upperboundary.mark(boundaries, 1)
#ds = Measure('ds')[boundaries]

# Set parameter values
dt = 0.01
T = 3
nu = 0.01 # since after rescaling nu is eps-indep, we can use nu1 = nu2 = nu3 = nu
eps = 1
theta1 = 1 # constant wind traction
theta2 = 1
alpha = 1
beta = 1

# Define time-dependent pressure boundary condition
#p_in = Expression("sin(3.0*t)", t = 0.0, degree = 2)

# Define boundary conditions
# originals
#noslip  = DirichletBC(V, (0, 0, 0),
#                      "on_boundary && \
#                       (x[0] < DOLFIN_EPS | x[0] > 1.0 - DOLFIN_EPS | x[1] > 1.0 - DOLFIN_EPS | x[1] < DOLFIN_EPS)")
#inflow  = DirichletBC(Q, p_in, "on_boundary && (x[2] > 1.0 - DOLFIN_EPS)")
#outflow = DirichletBC(Q, 0, "on_boundary && (x[2] < DOLFIN_EPS)")
# new

noslipbasin = DirichletBC(V, (0, 0, 0), "on_boundary && x[2] < 1.0 - DOLFIN_EPS")
def Dirichlet_boundary(x, on_boundary):
    return on_boundary and \
           (x[2] < 1.0 - DOLFIN_EPS)


# bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)
# it sets the value arg2 to the elements marked with arg4 of the sub_domains structure arg3 in the space arg1.
bcu = [noslipbasin]
bcp = []

# Create functions
u_prev = Function(V)
u_prev1, u_prev2, u_prev3 = split(u_prev)
u_next = Function(V)
u_next1, u_next2, u_next3 = split(u_next)
p_prev = Function(Q)
p_next = Function(Q)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0, 0))
U = 0.5 * (u_prev + u)

### in all 3 steps, the unknown function is
### denoted by u and p, whose type is by definition "TrialFunction"

# Tentative velocity step
# knowing u_prev, p_prev, we GET: u, ie u^*, the tentative velocity
F1 = (1/k)*( (u1 - u_prev1)*v1 + (u2 - u_prev2)*v2 + eps*eps*(u3 - u_prev3)*v3 ) * dx + \
     u_prev1 * inner(u_prev, grad(v1)) * dx - u_prev2 * inner(u_prev, grad(v2)) * dx + \
     eps*eps*inner(u_prev, grad(u_prev3)) * v3 * dx + \
     - alpha * u_prev2 * v1 * dx + alpha * u_prev1 * v2 * dx + \
     inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx + eps*eps*inner(grad(u3),grad(v3)) * dx + \
     eps * beta * u_prev3 * v1 * dx - eps * beta * u_prev1 * v3 * dx + \
     inner(grad(p_prev), v) * dx - inner(f, v) * dx
#     - theta1 * v1 * ds(1) - theta2 * v2 * ds(1)
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
# this is where we GET p, ie p^*. from the previous
# step we solve for u^*, save it in u_next, and so at this
# point u_next contains u^* (see later at the time steps)
a2 = inner(grad(p), grad(q))*dx
L2 = inner(grad(p_prev), grad(q))*dx - (1/k)*div(u_next)*q*dx

# Velocity update
# we know u_next (ie the tentative velocity u^*) and p_next ("p^*"), here we GET u.
a3 = inner(u, v)*dx
L3 = inner(u_next, v)*dx - k*inner(grad(p_next - p_prev), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:

    # Update pressure boundary condition
    #p_in.t = t

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

# Plot solution
plt.figure()
plot(p_next, title="Pressure")

plt.figure()
plot(u_next, title="Velocity")

plt.show()
