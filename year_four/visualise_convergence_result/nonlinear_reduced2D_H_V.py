# The anisotropic model works nicely with the (2,2) degree scenario, the hydrostatic scheme works for (2,1).
# The former does not really work for (2,1) as it develops strange unnatural layers in the pressure.
# The latter does not work for (2,2) as the Newton iterations do not converge --- probably because we do not have the first derivative of u3 in the scheme and u3 is second order in that case.

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import datetime
from numpy.random import rand
from dolfin import *
import argparse
import numpy

# Define constants

epsilon_lower_limit = 5e-04 #up 1.0e-07 c 5e-04
wind_shear_x = 10.0
theta = Constant((wind_shear_x, 0.0))
mu_1 = Constant(1.0)
mu_2 = Constant(1.0)

anisotropic_norm_u1_values = []
anisotropic_norm_u3_values = []
anisotropic_norm_p_values = []

anisotropic_interpolated_norm_u1_values = []
anisotropic_interpolated_norm_u3_values = []
anisotropic_interpolated_norm_p_values = []

interpolated_and_hydr_difference_norm_u1_values = []
interpolated_and_hydr_difference_norm_u3_values = []
interpolated_and_hydr_difference_norm_p_values = []

anis_and_hydr_difference_norm_u1_values = []
anis_and_hydr_difference_norm_p_values = []

timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultfolder", default="current_results_H_V", help="default: results, custom: name of results folder")
xargs = parser.parse_args(None)
resultsfolder = str(timestamp) + xargs.resultfolder + "/"

verbose = True

doHydrostatic = False
doAnisotropicLoop = False
doInitGuessHydro = False
doDegree1Anisopic = False
doRefineDomain = True
doErrorPlay = True

mesh = UnitSquareMesh(30, 30)

class nascent_delta(UserExpression):
    def __init__(self,eps,**kwargs):
        # Call superclass constructor with keyword arguments to properly
        # set up the instance:
        super().__init__(**kwargs)
        # Perform custom setup tasks for the subclass after that:
        self.eps = eps
    
    def eval(self, values, x):
        eps = self.eps
        # https://en.wikipedia.org/wiki/Cauchy_distribution#Multivariate_Cauchy_distribution
        # An example of a bivariate Cauchy distribution can be given by:
        values[0] = (1/(2 * pi)) * (eps / ( (x[0] - 0.5)**2 + (x[1] - 0.5)**2 + eps**2 )**(1.5) )
        #values[0] = eps**2 /pi/((x[0] - 0.5)**2 + (x[1] - 0.5)**2 + eps**2)

    def value_shape(self):
        return ()

# create a functionspace ((V_h, V_v), P) with given degree of V_v
def VP_functionspace(mesh, v_vert_deg):
    V_h = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 2) #horizontal velocity
    V_v = FiniteElement("Lagrange", mesh.ufl_cell(), degree = v_vert_deg) #vertical velocity
    V = V_h * V_v
    P = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 1) #pressure
    VP = FunctionSpace(mesh, V * P)
    return VP

# set boundary domains
class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)
upperboundary = UpperBoundary()

def boundaryconditions(VP):
    noslipbasin = DirichletBC(VP.sub(0), Constant((0, 0)), "on_boundary && x[1] < 1.0 - DOLFIN_EPS")
    zerotopvertical = DirichletBC(VP.sub(0).sub(1), 0, "on_boundary && x[1] > 1.0 - DOLFIN_EPS")
    bcu = [noslipbasin, zerotopvertical]
    return bcu

def writedifference(degree_anis, degree_hydr):
    np.savetxt(resultsfolder + "anisotropic_norm_u1_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", anisotropic_norm_u1_values)
    np.savetxt(resultsfolder + "anisotropic_norm_u3_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", anisotropic_norm_u3_values)
    np.savetxt(resultsfolder + "anisotropic_norm_p_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", anisotropic_norm_p_values)
    np.savetxt(resultsfolder + "anisotropic_interpolated_norm_u1_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", anisotropic_interpolated_norm_u1_values)
    np.savetxt(resultsfolder + "anisotropic_interpolated_norm_u3_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", anisotropic_interpolated_norm_u3_values)
    np.savetxt(resultsfolder + "anisotropic_interpolated_norm_p_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", anisotropic_interpolated_norm_p_values)
    np.savetxt(resultsfolder + "interpolated_and_hydr_difference_norm_u1_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", interpolated_and_hydr_difference_norm_u1_values)
    np.savetxt(resultsfolder + "interpolated_and_hydr_difference_norm_u3_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", interpolated_and_hydr_difference_norm_u3_values)
    np.savetxt(resultsfolder + "interpolated_and_hydr_difference_norm_p_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", interpolated_and_hydr_difference_norm_p_values)
    np.savetxt(resultsfolder + "anis_and_hydr_difference_norm_u1_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", anis_and_hydr_difference_norm_u1_values)
    np.savetxt(resultsfolder + "anis_and_hydr_difference_norm_p_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", anis_and_hydr_difference_norm_p_values)

def hydrostatic_solver(VP, up_, vertical_velocity_degree, mesh_h):

    nr_cells = mesh_h.num_cells()

    boundaries = MeshFunction("size_t", mesh_h, mesh_h.topology().dim() - 1)
    boundaries.set_all(0)
    upperboundary.mark(boundaries, 1)
    ds = Measure('ds')[boundaries]
    
    up = TrialFunction(VP)
    u,p = split(up) # u,p are "trial function" type (special to FEniCS)
    u1, u3 = split(u)
    (v, q) = TestFunctions(VP)
    v1, v3 = split(v)

    (u_, p_) = up_.split(True)
    (u1_, u3_) = u_.split(True)
    
    # the hydrostatic weak formulation without an initial guess (for now) is constructed with the vertical velocity space being of degree 1 and the additional constraint p.dx(1) * q.dx(1) * dx representing that we have a hydrostatic pressure. using a lower degree for the vertical velocities for the case of the primitive equations come from the article of Danilov, Gennady, Schroter, 2002 (even though they use elementwise constant representations)
    F = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx - p * div(v) * dx + q * div(u) * dx + p.dx(1) * q.dx(1) * dx - inner(theta, v) * ds(1)
    
    F = action(F, up_)
    J  = derivative(F, up_, up)
    
    # nonlinear solver for the velocity and pressure
    bcu = boundaryconditions(VP)
    problem = NonlinearVariationalProblem(F, up_, bcu, J)
    solver  = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1e-3
    prm['newton_solver']['relative_tolerance'] = 1e-2
    prm['newton_solver']['maximum_iterations'] = 5
    solver.solve()

    # from now on we process the data (note the usage of u,p as auxilliary variables of "function" type
    (u,p) = up_.split(True)
    (u1, u3) = u.split(True)
    
    ufile_pvd_hydr = File(resultsfolder + "velocity/velocity_hydr_degree" + str(vertical_velocity_degree) + "_nr_cells_" + str(nr_cells) + ".pvd")
    ufile_pvd_hydr << u
    pfile_pvd_hydr = File(resultsfolder + "pressure/pressure_hydr_degree" + str(vertical_velocity_degree) + "_nr_cells_" + str(nr_cells) + ".pvd")
    pfile_pvd_hydr << p
    
    hydrostatic_values = []
    hydrostatic_values.append(u1.vector().norm("l2"))
    hydrostatic_values.append(u3.vector().norm("l2"))
    hydrostatic_values.append(p.vector().norm("l2"))
    np.savetxt(resultsfolder + "hydrostatic_values_degree_" + str(vertical_velocity_degree)+ ".txt", hydrostatic_values)
    
    # concentration
    C = FiniteElement("Lagrange", mesh_h.ufl_cell(), degree = 2)
    C = FunctionSpace(mesh_h, C)
    zerotop_concentration = DirichletBC(C, 0, "on_boundary && x[1] > 1 - DOLFIN_EPS")
    bcc = [zerotop_concentration]
    c = TrialFunction(C)
    d = TestFunction(C)
    c_sol = Function(C)
    
    a = inner(u, grad(c)) * d * dx + (mu_1 * c.dx(0) * d.dx(0) + mu_2 * c.dx(1) * d.dx(1))  * dx - inner(c.dx(1), d.dx(1)) * ds(1)
    # linear solver for the concentration
    L = Constant(0) * d * dx
    
    A, b = assemble_system(a, L, bcc)
    
    delta = PointSource(C, Point(0.5, 0.5), 1)
    delta.apply(b)
    
    solver = KrylovSolver('gmres', 'ilu')
    solver.solve(A, c_sol.vector(), b)
    cfile_pvd_hydr = File(resultsfolder + "concentration/concentration_hydr_degree" + str(vertical_velocity_degree) + "_nr_cells_" + str(nr_cells) + ".pvd")
    cfile_pvd_hydr << c_sol
    print("HYDR. c: %.15g" % c_sol.vector().norm("l2"))
    
    return up_
    
def anisotropic_solver(VP, eps, vertical_velocity_degree, mesh_h):

    nr_cells = mesh_h.num_cells()

    boundaries = MeshFunction("size_t", mesh_h, mesh_h.topology().dim() - 1)
    boundaries.set_all(0)
    upperboundary.mark(boundaries, 1)
    ds = Measure('ds')[boundaries]

    up = TrialFunction(VP)
    u,p = split(up)
    u1, u3 = split(u)
    (v, q) = TestFunctions(VP)
    v1, v3 = split(v)

    up_ = Function(VP)
    (u_, p_) = split(up_)
    (u1_, u3_) = split(u_)

    # the anisotropic weak formulation is created using the Taylor-Hood elements, the vertical velocity is from a quadratic space. Using a degree 1 vertical velocity space in the anisotropic case we have a strange layered unnatural pressure.
    F = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx + eps*eps*inner(u, grad(u3)) * v3 * dx + eps*eps*inner(grad(u3),grad(v3)) * dx - p * div(v) * dx + q * div(u) * dx - inner(theta, v) * ds(1)
    
    F = action(F, up_)
    J = derivative(F, up_)

    bcu = boundaryconditions(VP)
    problem = NonlinearVariationalProblem(F, up_, bcu, J)
    solver  = NonlinearVariationalSolver(problem)
    solver.solve()

    (u,p) = up_.split(True)

    ufile_pvd_anis = File(resultsfolder + "velocity/velocity_anis_degree" + str(vertical_velocity_degree) + "_eps_" + str(eps) + "_nr_cells_" + str(nr_cells) + ".pvd")
    pfile_pvd_anis = File(resultsfolder + "pressure/pressure_anis_degree" + str(vertical_velocity_degree) + "_eps_" + str(eps) + "_nr_cells_" + str(nr_cells) + ".pvd")
    ufile_pvd_anis << u
    pfile_pvd_anis << p
    
    C = FiniteElement("Lagrange", mesh_h.ufl_cell(), degree = 2)
    C = FunctionSpace(mesh_h, C)
    zerotop_concentration = DirichletBC(C, 0, "on_boundary && x[1] > 1 - DOLFIN_EPS")
    bcc = [zerotop_concentration]
    c = TrialFunction(C)
    d = TestFunction(C)
    c_sol = Function(C)
    
    a = inner(u, grad(c)) * d * dx + (mu_1 * c.dx(0) * d.dx(0) + mu_2 * c.dx(1) * d.dx(1))  * dx - inner(c.dx(1), d.dx(1)) * ds(1)
    
    # explanation of the degree parameter: https://fenicsproject.discourse.group/t/how-to-define-source-term-function/1893, Scan_29_Nov_2019.pdf.
    # the main idea is that "degree" is a built-in parameter in this class, we do not need to "create" it. it gets defined through the call,
    # and it probably has an effect on the degree of approximation in terms of what degree is used in the \int s * phi dx integral
    # where phi is the test function, s is the source, and in the background (probably) some sort of quadrature is used to approximate this integral.
    # the degree of the quadrature is this degree, probably, or something similar.
    # Also: if degree is set to a high value, e.g. degree = 20, a warning message comes from fenics:
    # "WARNING: The number of integration points for each cell will be: 144"
    # i.e. this degree variable is responsable for the number of integration points
    nascent_delta_instance = nascent_delta(eps, degree = 10)
    L = inner(nascent_delta_instance, d) * dx
    
    A, b = assemble_system(a, L, bcc)
    
    solver = KrylovSolver('gmres', 'ilu')
    solver.solve(A, c_sol.vector(), b)
    
    cfile_pvd_anis = File(resultsfolder + "concentration/concentration_anis_degree" + str(vertical_velocity_degree) + "_eps_" + str(eps) + "_nr_cells_" + str(nr_cells) + ".pvd")
    cfile_pvd_anis << c_sol
    print(eps)
    print("ANIS. c: %.15g" % c_sol.vector().norm("l2"))
    
    return up_

def difference_info(eps, up_sol_anis_eps, VPA, up_sol_hydr, VPH):

    (u, p) = up_sol_anis_eps.split(True)
    (u1, u3) = u.split(True)
    
    anisotropic_norm_u1_values.append(u1.vector().norm("l2"))
    anisotropic_norm_u3_values.append(u3.vector().norm("l2"))
    anisotropic_norm_p_values.append(p.vector().norm("l2"))
    
    up_interpolate_hydr = Function(VPH)
    up_interpolate_hydr = interpolate(up_sol_anis_eps, VPH)
    (u_interpolate_hydr, p_interpolate_hydr) = up_interpolate_hydr.split(True)
    (u1_interpolate_hydr, u3_interpolate_hydr) = u_interpolate_hydr.split(True)
    
    anisotropic_interpolated_norm_u1_values.append(u1_interpolate_hydr.vector().norm("l2"))
    anisotropic_interpolated_norm_u3_values.append(u3_interpolate_hydr.vector().norm("l2"))
    anisotropic_interpolated_norm_p_values.append(p_interpolate_hydr.vector().norm("l2"))
    
    (u_sol_hydr, p_sol_hydr) = up_sol_hydr.split(True)
    (u1_sol_hydr, u3_sol_hydr) = u_sol_hydr.split(True)
    
    interpolated_and_hydr_difference_norm_u1_values.append((u1_interpolate_hydr.vector() - u1_sol_hydr.vector()).norm("l2"))
    interpolated_and_hydr_difference_norm_u3_values.append((u3_interpolate_hydr.vector() - u3_sol_hydr.vector()).norm("l2"))
    interpolated_and_hydr_difference_norm_p_values.append((p_interpolate_hydr.vector() - p_sol_hydr.vector()).norm("l2"))
    
    anis_and_hydr_difference_norm_u1_values.append((u1.vector() - u1_sol_hydr.vector()).norm("l2"))
    # this does not make sense for different degree spaces
    #anis_and_hydr_difference_norm_u3_values.append((u3.vector() - u3_sol_hydr.vector()).norm("l2"))
    anis_and_hydr_difference_norm_p_values.append((p.vector() - p_sol_hydr.vector()).norm("l2"))
    
    if (verbose):
        print(eps)
        print("Anistropic. u: %.15g" % u.vector().norm("l2"))
        print("Anistropic. u1: %.15g" % u1.vector().norm("l2"))
        print("Anistropic. u3: %.15g" % u3.vector().norm("l2"))
        print("Anistropic. p: %.15g" % p.vector().norm("l2"))
        print("Anistropic Interpolated. u: %.15g" % u_interpolate_hydr.vector().norm("l2"))
        print("Anistropic Interpolated. u1: %.15g" % u1_interpolate_hydr.vector().norm("l2"))
        print("Anistropic Interpolated. u3: %.15g" % u3_interpolate_hydr.vector().norm("l2"))
        print("Anistropic Interpolated. p: %.15g" % p_interpolate_hydr.vector().norm("l2"))
        print("Anistropic Interpolated - Hydrostatic. u: %.15g" % (u_interpolate_hydr.vector() - u_sol_hydr.vector()).norm("l2"))
        print("Anistropic Interpolated - Hydrostatic. u1: %.15g" % (u1_interpolate_hydr.vector() - u1_sol_hydr.vector()).norm("l2"))
        print("Anistropic Interpolated - Hydrostatic. u3: %.15g" % (u3_interpolate_hydr.vector() - u3_sol_hydr.vector()).norm("l2"))
        print("Anistropic Interpolated - Hydrostatic. p: %.15g" % (p_interpolate_hydr.vector() - p_sol_hydr.vector()).norm("l2"))
    

def refine_domain(cells_division, eps):

    mesh_h = UnitSquareMesh(cells_division, cells_division)
    print("h: %.15g" % cells_division)
    
    vertical_velocity_degree_anis = 2
    VP = VP_functionspace(mesh_h, vertical_velocity_degree_anis)
    up_sol_anis_eps = Function(VP)
    up_sol_anis_eps = anisotropic_solver(VP, eps, vertical_velocity_degree_anis, mesh_h)
    
    vertical_velocity_degree_hydr = 1
    VPH = VP_functionspace(mesh_h, vertical_velocity_degree_hydr)
    up_ = Function(VPH)
    up_sol_hydr = hydrostatic_solver(VPH, up_, vertical_velocity_degree_hydr, mesh_h)

# **********************************************
# *** Define hydrostatic variational problem ***
# **********************************************

if (doHydrostatic):
    # hydrostatic model solved without initial guess for degree 1 vertical velocity space
    vertical_velocity_degree_hydr = 1
    VPH = VP_functionspace(mesh, vertical_velocity_degree_hydr)
    up_ = Function(VPH) #initial guess for the Newton solver if filled, otherwise blank and start by default
    up_sol_hydr = hydrostatic_solver(VPH, up_, vertical_velocity_degree_hydr, mesh)

# **********************************************
# *** Define anisotropic variational problem ***
# **********************************************
if (doAnisotropicLoop):
    eps = 1.0
    vertical_velocity_degree_anis = 2
    VP = VP_functionspace(mesh, vertical_velocity_degree_anis)
    up_sol_anis_eps = Function(VP)

    while eps > epsilon_lower_limit:
    
        up_sol_anis_eps = anisotropic_solver(VP, eps, vertical_velocity_degree_anis, mesh)
        difference_info(eps, up_sol_anis_eps, VP, up_sol_hydr, VPH)
        eps = eps / 2.0
    
    writedifference(vertical_velocity_degree_anis, vertical_velocity_degree_hydr)

# **********************************************
# *** degree 2 for the hydrostatic weak form ***
# **********************************************
if (doAnisotropicLoop and doInitGuessHydro):
    # hydrostatic model solved with initial guess for degree 2 vertical velocity space
    vertical_velocity_degree_hydr = 2
    up_sol_hydr = hydrostatic_solver(VP, up_sol_anis_eps, vertical_velocity_degree_hydr, mesh)

# **************************************************************
# *** Define anisotropic variational problem  with degree = 1 **
# **************************************************************
if (doDegree1Anisopic):
    eps = 1.0
    vertical_velocity_degree_anis = 1
    VP = VP_functionspace(mesh, vertical_velocity_degree_anis)

    while eps > epsilon_lower_limit:
    
        anisotropic_solver(VP, eps, vertical_velocity_degree_anis, mesh)
        eps = eps / 2.0

# **************************************************************
# ************************** Loop in h *************************
# **************************************************************

if (doRefineDomain):
    print("Loop in h.")

    eps = 5e-04
    nx_exp = 4
    nx = 2 ** nx_exp # to control the number of cells, UnitSquareMesh(nx, nx)
    while nx < 2 ** 8:
        refine_domain(nx, eps)
        nx = 2 * nx

if (doErrorPlay):

    # the following setup provides a function that is divergence-free and works with periodic BCs.
    # in other words, solving for periodic BCs with this particular f, this is the solution.
    
    # the following works for eps being 1.
    u_exact1 = Expression('sin(2*pi*x[0])*cos(2*pi*x[1])', degree = 3)
    u_exact2 = Expression('-cos(2*pi*x[0])*sin(2*pi*x[1])', degree = 3)
    p_exact = Expression('(1-x[1]))+x[0]', degree = 3)
    
    # substituting these three functions into the strong form, for the right-hand side we get:
    f_1 = Expression('cos(2*pi*x[1])*(sin(2*pi*x[0]) + 2*pi*cos(2*pi*x[0])) + sin(2*pi*x[1])*(cos(2*pi*x[0]) + 2*pi*sin(2*pi*x[0])) + 4*pi*pi*2*sin(2*pi*x[0])*cos(2*pi*x[1]) + 1', degree = 3)
    f_2 = Expression('sin(2*pi*x[0])*(cos(2*pi*x[1]) + 2*pi*sin(2*pi*x[1])) + cos(2*pi*x[0])*(sin(2*pi*x[1]) + 2*pi*cos(2*pi*x[1])) - 4*pi*pi*2*cos(2*pi*x[0])*sin(2*pi*x[1]) - 1', degree = 3)
    
    # step2: prepare the solvers to have a forcing term
    
    # step3: define an alternative set of boundary conditions, set them as arguments so they can be chosen when calling a function.
    


# firstly, let's use u = sin(x)sin(z)

# *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- #

# improvement ideas:

# 0) add C

# i) the scheme depends epsilon-freely on the complete u. why do the Newton iterations not converge for the case when we have degree=2 for the vertical velocity? is it that the scheme contains only u3, but not grad(u3), and it is a 2-degree space? probably not in general, as for mesh(1,1) the method actually does converge.

# ii) is it possible to make it time-dependent?

# iii) 3D in space

# iv) periodicity in the x direction would make the domain into a tube with upper wind traction and fully x-directional circulation. Having a 0 y-directional velocity would be ok in itself, but with that there is not much to visualise as the scheme then does not depend on epsilon. so I think it is better to have the classical domain in order to make a point with the visualisation.

# v) print the Jacobian, makes sense for small mesh:

#J_mat = assemble(J)
#J_array = J_mat.array()
#np.savetxt("Jacobianmatrix_h2.txt", J_array)
#detJ = numpy.linalg.det(J_array)
#print("det Jacobian h 2:")
#print(detJ)


