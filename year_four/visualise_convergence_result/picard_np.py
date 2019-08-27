"""
FEniCS tutorial demo program:
Nonlinear Poisson equation with Dirichlet conditions
in x-direction and homogeneous Neumann (symmetry) conditions
in all other directions. The domain is the unit hypercube in
of a given dimension.

-div(q(u)*nabla_grad(u)) = 0,
u = 0 at x=0, u=1 at x=1, du/dn=0 at all other boundaries.
q(u) = (1+u)^m

Solution method: Picard iteration (successive substitutions).
"""

# call as fenics@b31a71fe5686:~/shared$ python3 picard_np.py 2 10 10 10
# where 2 is the degree, and the remaining list represents the divisions (its length becomes the dimension)

from dolfin import *
import numpy, sys

# Create mesh and define function space
degree = int(sys.argv[1])
divisions = [int(arg) for arg in sys.argv[2:]]
d = len(divisions)
print("d:")
print(d)
print("degree:")
print(degree)
domain_type = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
mesh = domain_type[d-1](*divisions)
V = FunctionSpace(mesh, 'Lagrange', degree)

# Define boundary conditions

tol = 1E-14
def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < tol

def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < tol

Gamma_0 = DirichletBC(V, Constant(0.0), left_boundary)
Gamma_1 = DirichletBC(V, Constant(1.0), right_boundary)
bcs = [Gamma_0, Gamma_1]

# Choice of nonlinear coefficient
m = 2

def q(u):
    return (1+u)**m

# Define variational problem for Picard iteration
u = TrialFunction(V)
v = TestFunction(V)
u_k = interpolate(Constant(0.0), V)  # previous (known) u
a = inner(q(u_k)*nabla_grad(u), nabla_grad(v))*dx
f = Constant(0.0)
L = f*v*dx

# Picard iterations
u = Function(V)     # new unknown function
eps = 1.0           # error measure ||u-u_k||
tol = 1.0E-5        # tolerance
iter = 0            # iteration counter
maxiter = 25        # max no of iterations allowed
while eps > tol and iter < maxiter:
    iter += 1
    solve(a == L, u, bcs)
    diff = u.vector().get_local() - u_k.vector().get_local()
    eps = numpy.linalg.norm(diff, ord=numpy.Inf)
    u_k.assign(u)   # update for next iteration

convergence = 'convergence after %d Picard iterations' % iter
if iter >= maxiter:
    convergence = 'no ' + convergence
print("convergence:")
print(convergence)

# Find max error
u_exact = Expression('pow((pow(2, 3.0)-1)*x[0] + 1, 1.0/3.0) - 1', degree = 3)
u_e = interpolate(u_exact, V)
diff = numpy.abs(u_e.vector().get_local() - u.vector().get_local()).max()
print("diff:")
print(diff)
