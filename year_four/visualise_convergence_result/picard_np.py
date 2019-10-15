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
u1 = TrialFunction(V)
v1 = TestFunction(V)
u1_k = interpolate(Constant(0.0), V)  # previous (known) u1
u2 = TrialFunction(V)
v2 = TestFunction(V)
u2_k = interpolate(Constant(0.0), V)  # previous (known) u2
a = inner(q(u1_k)*nabla_grad(u1), nabla_grad(v1))*dx + inner(q(u2_k)*nabla_grad(u2), nabla_grad(v2))*dx
f = Constant(0.0)
L = f*v1*dx + f*v2*dx

# Picard iterations
u1 = Function(V)     # new unknown function
u2 = Function(V)     # new unknown function
eps = 1.0           # error measure ||u-u_k||
tol = 1.0E-5        # tolerance
iter = 0            # iteration counter
maxiter = 25        # max no of iterations allowed
print("iteration starts:")
while eps > tol and iter < maxiter:
    iter += 1
    solve(a == L, u1, u2, bcs)
    diff = u1.vector().get_local() - u1_k.vector().get_local() + u2.vector().get_local() - u2_k.vector().get_local()
    eps = numpy.linalg.norm(diff, ord=numpy.Inf)
    u1_k.assign(u1)   # update for next iteration
    u2_k.assign(u2)   # update for next iteration

convergence = 'convergence after %d Picard iterations' % iter
if iter >= maxiter:
    convergence = 'no ' + convergence
print("convergence:")
print(convergence)

u1file = File("out1.pvd")
u1file << u1
u2file = File("out2.pvd")
u2file << u2
