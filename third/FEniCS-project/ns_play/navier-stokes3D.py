import matplotlib.pyplot as plt
from dolfin import *

parameters["std_out_all_processes"] = False;
mesh = UnitCubeMesh(5,5,5)

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
eps = 1.0
alpha = 1.0
beta = 1.0

noslipbasin = DirichletBC(V, (0, 0, 0), "on_boundary && x[2] < 1.0 - DOLFIN_EPS")

bcu = [noslipbasin]
bcp = []

u_prev = Function(V)
u_prev1, u_prev2, u_prev3 = split(u_prev)
u_next = Function(V)
u_next1, u_next2, u_next3 = split(u_next)
p_prev = Function(Q)
p_next = Function(Q)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0, 0))
theta = Constant((0.5, 0.5, 0))

F1 = (1/k)*( (u1 - u_prev1)*v1 + (u2 - u_prev2)*v2 + eps*eps*(u3 - u_prev3)*v3 ) * dx + \
     u_prev1 * inner(u_prev, grad(v1)) * dx - u_prev2 * inner(u_prev, grad(v2)) * dx + \
     eps*eps*inner(u_prev, grad(u_prev3)) * v3 * dx + \
     - alpha * u_prev2 * v1 * dx + alpha * u_prev1 * v2 * dx + \
     inner(grad(u1),grad(v1)) * dx + inner(grad(u2),grad(v2)) * dx + eps*eps*inner(grad(u3),grad(v3)) * dx + \
     eps * beta * u_prev3 * v1 * dx - eps * beta * u_prev1 * v3 * dx + \
     inner(grad(p_prev), v) * dx - inner(f, v) * dx + \
     - inner(theta, v) * ds(1)
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = inner(grad(p_prev), grad(q))*dx - (1/k)*div(u_next)*q*dx

# Velocity update
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
