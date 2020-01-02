# The anisotropic model works nicely with the (2,2) degree scenario, the hydrostatic scheme works for (2,1).
# The former does not really work for (2,1) as it develops strange unnatural layers in the pressure.
# The latter does not work for (2,2) as the Newton iterations do not converge --- probably because we do not have the first derivative of u3 in the scheme and u3 is second order in that case.

# BOUNDARY comments:
# 1.) the Dirichlet part of the boundary, \Gamma: you do not add the boundary terms on \Gamma to the F form itself as something like c.dx(1)*d*dx(\Gamma). instead, you define a DirichletBC condition, and then apply it to the Problem itself.
# 2.) on parts of the boundary where we have a DirichletBC defined, the test functions go to zero, thus an added boundary integral would not change anything
# 3.) in FEniCS the test functions go to zero on and only on the boundary section for which we have DirichletBC defined
# 4.) boundary integral terms in the F form should be used exactly for those boundary sections where we do not have a DirichletBC. And among these, where we do not have a Dirichlet condition, we either have a NeumannBC, or, we risk the problem being ill-posed.
# 5.) on a boundary section where we do not have a DirichletBC: in many problems the h value of the Neumann BC is zero, and the boundary term is therefore omitted; this case is sometimes referred to as the “do-nothing boundary condition”.
# 6.) something like grad(u)*v*dx(\Gamma) in itself for Neumann BC-s should not be used. you know that grad(u) = h for a given h on \Gamma, then you add h*v*dx(\Gamma) in the form F.

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import datetime
from numpy.random import rand
from dolfin import *
import argparse
import numpy
from math import log
import os

import write_plot_tools
import global_lists

global_lists.global_lists_init()

# Define constants

epsilon_lower_limit = 1.0e-07 #up 1.0e-07 c 5e-04

log_errorvalues_L2_u1 = []
log_errorvalues_L2_u3 = []
log_errorvalues_L2_p = []
log_errorvalues_L2_c = []
log_errorvalues_H1_u1 = []
log_errorvalues_H1_u3 = []
log_errorvalues_H1_p = []
log_errorvalues_H1_c = []
nxvalues = []

errorvalues = []
eocvalues = []

timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultfolder", default="current_results_H_V", help="default: results, custom: name of results folder")
xargs = parser.parse_args(None)
resultsfolder = str(timestamp) + xargs.resultfolder + "/"

verbose = True

doHydrostatic = True
doAnisotropicLoop = True
doInitGuessHydro = True
doDegree1Anisopic = True
doErrorCalc = True

mesh = UnitSquareMesh(30, 30)

# set boundary domains

class LateralBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and (near(x[0], 0.0, tol) or near(x[0], 1.0, tol))

class UpperBottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and (near(x[1], 0.0, tol) or near(x[1], 1.0, tol))

class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[1], 1.0)
    
class LowerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[1], 0.0, tol)
        
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], 0.0, tol)
 
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], 1.0, tol)
        
def concentration_BCs_pd(id, C):

    zerotop_concentration = DirichletBC(C, 0.0, "on_boundary && x[1] > 1 - DOLFIN_EPS")
    zerobc = DirichletBC(C, 0.0, "on_boundary")

    if id == 0:
        return [zerotop_concentration]
    elif id == 1 :
        return [zerobc]
    elif id == 2:
        return [zerobc]
    elif id == 3:
        return [zerobc]
    elif id == 4:
        return [zerobc]
    else:
        return []

def boundaryconditions_pd(id, VP):

    lateral_boundary = LateralBoundary()
    upper_bottom_boundary = UpperBottomBoundary()
    upper_boundary = UpperBoundary()
    lower_boundary = LowerBoundary()
    left_boundary = LeftBoundary()
    right_boundary = RightBoundary()

    if id == 0:
    
        noslipbasin = DirichletBC(VP.sub(0), Constant((0, 0)), "on_boundary && x[1] < 1.0 - DOLFIN_EPS")
        zerotopvertical = DirichletBC(VP.sub(0).sub(1), 0, "on_boundary && x[1] > 1.0 - DOLFIN_EPS")
        bcu = [noslipbasin, zerotopvertical]
        return bcu

    elif id == 1:
    
        zeroboundaryu = DirichletBC(VP.sub(0), Constant((0, 0)), "on_boundary")
        bcuerrest = [zeroboundaryu]
        return bcuerrest
    
    elif id == 2:
    
        zeroboundaryu = DirichletBC(VP.sub(0), Constant((0, 0)), "on_boundary")
        bcuerrest = [zeroboundaryu]
        return bcuerrest
    
    elif id == 3:

        zeroleftu1 = DirichletBC(VP.sub(0).sub(0), 0.0, left_boundary)
        onerightu1 = DirichletBC(VP.sub(0).sub(0), 1.0, right_boundary)
        zeroloweru3 = DirichletBC(VP.sub(0).sub(1), 0.0, lower_boundary)
        minusoneupperu3 = DirichletBC(VP.sub(0).sub(1), -1.0, upper_boundary)

        p_lateral = Expression('0', degree = 3)
        p_upperbottom = Expression('0', degree = 3)
        pressureBClateral = DirichletBC(VP.sub(1), p_lateral, lateral_boundary)
        pressureBCupperbottom = DirichletBC(VP.sub(1), p_upperbottom, upper_bottom_boundary)

        bcuerrest = [zeroleftu1, onerightu1, zeroloweru3, minusoneupperu3, pressureBClateral, pressureBCupperbottom]
        return bcuerrest
    
    elif id == 4 :
    
        zerolateralu1 = DirichletBC(VP.sub(0).sub(0), 0.0, lateral_boundary)
        zerotopbottomu3 = DirichletBC(VP.sub(0).sub(1), 0.0, upper_bottom_boundary)
        p_lateral = Expression('0', degree = 3)
        p_upperbottom = Expression('0', degree = 3)
        pressureBClateral = DirichletBC(VP.sub(1), p_lateral, lateral_boundary)
        pressureBCupperbottom = DirichletBC(VP.sub(1), p_upperbottom, upper_bottom_boundary)
        bcuerrest = [zerolateralu1, zerotopbottomu3, pressureBClateral, pressureBCupperbottom]
        return bcuerrest
        
    else:
        return []

class ProblemData(UserExpression):
    def __init__(self, id, u1_exact, u3_exact, p_exact, f1, f3, c_exact):
        self.id = id
        self.u1_exact = u1_exact
        self.u3_exact = u3_exact
        self.p_exact = p_exact
        self.f1 = f1
        self.f3 = f3
        self.c_exact = c_exact

# for the symbolic computations providing the forcing terms using the exact solutions we use sympy-1.4: Documents/sympy-1.4/examples/beginner/differentiation.py
# PROBLEM 0
# the actual one we solve

# PROBLEM 1
id = 1
u1_exact = Expression('0', degree = 5)
u3_exact = Expression('0', degree = 5)
p_exact = Expression('0', degree = 5)
f1 = Expression('0.0', degree = 5)
f3 = Expression('0.0', degree = 5)
c_exact = Expression('0.0', degree = 5)
problem_data1 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact)

# PROBLEM 2
id = 2
u1_exact = Expression('0', degree = 5)
u3_exact = Expression('0', degree = 5)
p_exact = Expression('0', degree = 5)
f1 = Expression('0.0', degree = 5)
f3 = Expression('0.0', degree = 5)
c_exact = Expression('x[0]*(1-x[0])*x[1]*(1-x[1])', degree = 5)
problem_data2 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact)

# PROBLEM 3
id = 3
u1_exact = Expression('x[0]', degree = 5)
u3_exact = Expression('-x[1]', degree = 5)
p_exact = Expression('0', degree = 5)
f1 = Expression('x[0]', degree = 5)
f3 = Expression('x[1]', degree = 5)
c_exact = Expression('x[0]*(1-x[0])*x[1]*(1-x[1])', degree = 5)
problem_data3 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact)

# PROBLEM 4
id = 4
u1_exact = Expression('sin(2*pi*x[0])*cos(2*pi*x[1])', degree = 5)
u3_exact = Expression('-cos(2*pi*x[0])*sin(2*pi*x[1])', degree = 5)
p_exact = Expression('0', degree = 5)
f1 = Expression('2*pi*sin(2*pi*x[0])*sin(2*pi*x[1])*sin(2*pi*x[1])*cos(2*pi*x[0]) + 2*pi*sin(2*pi*x[0])*cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[1]) + 8*pi*pi*sin(2*pi*x[0])*cos(2*pi*x[1])', degree = 5)
f3 = Expression('2*pi*sin(2*pi*x[0])*sin(2*pi*x[0])*sin(2*pi*x[1])*cos(2*pi*x[1]) + 2*pi*sin(2*pi*x[1])*cos(2*pi*x[0])*cos(2*pi*x[0])*cos(2*pi*x[1]) - 8*pi*pi*sin(2*pi*x[1])*cos(2*pi*x[0])', degree = 5)
c_exact = Expression('(1-x[0])*(1-x[1])*sin(2*pi*x[1])*sin(2*pi*x[0])', degree = 5)
problem_data4 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact)

problem_data_list = [problem_data1, problem_data2, problem_data3, problem_data4]

class anis_c_source(UserExpression):
    def __init__(self,eps,id,**kwargs):
        # Call superclass constructor with keyword arguments to properly
        # set up the instance:
        super().__init__(**kwargs)
        # Perform custom setup tasks for the subclass after that:
        self.eps = eps
        self.id = id

    def eval(self, values, x):
        eps = self.eps
        id = self.id
        
        if id == 0:
            # id = 0, original case
            # https://en.wikipedia.org/wiki/Cauchy_distribution#Multivariate_Cauchy_distribution
            # An example of a bivariate Cauchy distribution can be given by:
            values[0] = (1/(2 * pi)) * (eps / ( (x[0] - 0.5)**2 + (x[1] - 0.5)**2 + eps**2 )**(1.5) )
        
        elif id == 1:
            values[0] = 0
        
        elif id == 2:
            values[0] = 2*x[0]*(1-x[0]) + 2*x[1]*(1-x[1])
        
        elif id == 3:
            values[0] = 2*x[0]*(-x[0] + 1) + x[0]*(-x[0]*x[1]*(-x[1] + 1) + x[1]*(-x[0] + 1)*(-x[1] + 1)) + 2*x[1]*(-x[1] + 1) - x[1]*(-x[0]*x[1]*(-x[0] + 1) + x[0]*(-x[0] + 1)*(-x[1] + 1))
        
        elif id == 4:
            # test case 4
            values[0] = 8*pi**2*(-x[0] + 1)*(-x[1] + 1)*sin(2*pi*x[0])*sin(2*pi*x[1]) + 4*pi*(-x[0] + 1)*sin(2*pi*x[0])*cos(2*pi*x[1]) + 4*pi*(-x[1] + 1)*sin(2*pi*x[1])*cos(2*pi*x[0]) - (2*pi*(-x[0] + 1)*(-x[1] + 1)*sin(2*pi*x[0])*cos(2*pi*x[1]) - (-x[0] + 1)*sin(2*pi*x[0])*sin(2*pi*x[1]))*sin(2*pi*x[1])*cos(2*pi*x[0]) + (2*pi*(-x[0] + 1)*(-x[1] + 1)*sin(2*pi*x[1])*cos(2*pi*x[0]) - (-x[1] + 1)*sin(2*pi*x[0])*sin(2*pi*x[1]))*sin(2*pi*x[0])*cos(2*pi*x[1])
        
        else:
            values[0] = 0

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

def calculate_errorvalues(problem_data, upc_sol_anis_eps, nx):

    up_sol_anis_eps = upc_sol_anis_eps[0]
    c = upc_sol_anis_eps[1]
    (u, p) = up_sol_anis_eps.split(True)
    (u1, u3) = u.split(True)
    Eu1 = errornorm(problem_data.u1_exact, u1, norm_type='L2')
    Eu3 = errornorm(problem_data.u3_exact, u3, norm_type='L2')
    Ep = errornorm(problem_data.p_exact, p, norm_type='L2')
    Ec = errornorm(problem_data.c_exact, c, norm_type='L2', mesh = UnitSquareMesh(nx, nx))
    
    Eu1_H = errornorm(problem_data.u1_exact, u1, norm_type='H1')
    Eu3_H = errornorm(problem_data.u3_exact, u3, norm_type='H1')
    Ep_H = errornorm(problem_data.p_exact, p, norm_type='H1')
    Ec_H = errornorm(problem_data.c_exact, c, norm_type='H1', mesh = UnitSquareMesh(nx, nx))
    errorvalues.append(nx)
    errorvalues.append(Eu1)
    errorvalues.append(Eu3)
    errorvalues.append(Ep)
    errorvalues.append(Ec)
    errorvalues.append(Eu1_H)
    errorvalues.append(Eu3_H)
    errorvalues.append(Ep_H)
    errorvalues.append(Ec_H)
    
    nxvalues.append(nx)
    log_errorvalues_L2_u1.append(log(max(Eu1, DOLFIN_EPS)))
    log_errorvalues_L2_u3.append(log(max(Eu3, DOLFIN_EPS)))
    log_errorvalues_L2_p.append(log(max(Ep, DOLFIN_EPS)))
    log_errorvalues_L2_c.append(log(max(Ec, DOLFIN_EPS)))
    log_errorvalues_H1_u1.append(log(max(Eu1_H, DOLFIN_EPS)))
    log_errorvalues_H1_u3.append(log(max(Eu3_H, DOLFIN_EPS)))
    log_errorvalues_H1_p.append(log(max(Ep_H, DOLFIN_EPS)))
    log_errorvalues_H1_c.append(log(max(Ec_H, DOLFIN_EPS)))
    
    error_L2 = [Eu1, Eu3, Ep, Ec]
    
    return error_L2

def plot_exact_solutions(resultsfolder, nx, problem_data, foldermarker):

    mesh_h = UnitSquareMesh(nx, nx)
    W = FunctionSpace(mesh_h, 'Lagrange', 2)
    P = FunctionSpace(mesh_h, 'Lagrange', 1)
    C = FunctionSpace(mesh_h, 'Lagrange', 1)
    u1_W = interpolate(problem_data.u1_exact, W)
    u3_W = interpolate(problem_data.u3_exact, W)
    p_P = interpolate(problem_data.p_exact, P)
    c_C = interpolate(problem_data.c_exact, C)
    u1_exact_plot = File(resultsfolder + "velocity" + foldermarker + "/u1_exact_nx_" + str(nx) + ".pvd")
    u1_exact_plot << u1_W
    u3_exact_plot = File(resultsfolder + "velocity" + foldermarker + "/u3_exact_nx_" + str(nx) + ".pvd")
    u3_exact_plot << u3_W
    p_exact_plot = File(resultsfolder + "pressure" + foldermarker + "/p_exact_nx_" + str(nx) + ".pvd")
    p_exact_plot << p_P
    c_exact_plot = File(resultsfolder + "concentration" + foldermarker + "/c_exact_nx_" + str(nx) + ".pvd")
    c_exact_plot << c_C

def hydrostatic_solver(VP, up_, vertical_velocity_degree, mesh_h, f1, f3, theta, bcu):

    nr_cells = mesh_h.num_cells()

    upperboundary = UpperBoundary()
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
    F = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx - p * div(v) * dx + q * div(u) * dx + p.dx(1) * q.dx(1) * dx - f1 * v1 * dx - f3 * v3 * dx - inner(theta, v) * ds(1)
    
    F = action(F, up_)
    J = derivative(F, up_, up)
    
    # nonlinear solver for the velocity and pressure
    problem = NonlinearVariationalProblem(F, up_, bcu, J)
    solver  = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1e-9
    prm['newton_solver']['relative_tolerance'] = 1e-9
    prm['newton_solver']['maximum_iterations'] = 5
    solver.solve()

    # from now on we process the data (note the usage of u,p as auxilliary variables of "function" type
    (u,p) = up_.split(True)
    (u1, u3) = u.split(True)
    
    ufile_pvd_hydr = File(resultsfolder + "velocity_hydr/velocity_hydr_degree" + str(vertical_velocity_degree) + "_nr_cells_" + str(nr_cells) + ".pvd")
    ufile_pvd_hydr << u
    pfile_pvd_hydr = File(resultsfolder + "pressure_hydr/pressure_hydr_degree" + str(vertical_velocity_degree) + "_nr_cells_" + str(nr_cells) + ".pvd")
    pfile_pvd_hydr << p
    
    hydrostatic_values = []
    hydrostatic_values.append(u1.vector().norm("l2"))
    hydrostatic_values.append(u3.vector().norm("l2"))
    hydrostatic_values.append(p.vector().norm("l2"))
    np.savetxt(resultsfolder + "hydrostatic_values_degree_" + str(vertical_velocity_degree)+ ".txt", hydrostatic_values)
    
    # concentration
    C = FiniteElement("Lagrange", mesh_h.ufl_cell(), degree = 1)
    C = FunctionSpace(mesh_h, C)
    zerotop_concentration = DirichletBC(C, 0, "on_boundary && x[1] > 1 - DOLFIN_EPS")
    bcc = [zerotop_concentration]
    c = TrialFunction(C)
    d = TestFunction(C)
    c_sol = Function(C)
    
    # works for 0 BC concentation. otherwise, -c.dx(0)*d*ds(0)-c.dx(1)*d*ds(0) should be added, but for some reason the concentration error convergence values are not great for those cases, even if the velocity is very close
    a = inner(u, grad(c)) * d * dx + (c.dx(0) * d.dx(0) + c.dx(1) * d.dx(1)) * dx
    
    # linear solver for the concentration
    L = Constant(0) * d * dx
    
    A, b = assemble_system(a, L, bcc)
    
    delta = PointSource(C, Point(0.5, 0.5), 1)
    delta.apply(b)
    
    solver = KrylovSolver('gmres', 'ilu')
    solver.solve(A, c_sol.vector(), b)
    cfile_pvd_hydr = File(resultsfolder + "concentration_hydr/concentration_hydr_degree" + str(vertical_velocity_degree) + "_nr_cells_" + str(nr_cells) + ".pvd")
    cfile_pvd_hydr << c_sol
    print("HYDR. c: %.15g" % c_sol.vector().norm("l2"))
    
    return [up_, c_sol]
    
def anisotropic_solver(VP, eps, vertical_velocity_degree, mesh_h, f1, f3, theta, bcu, foldermarker, id):

    nr_cells = mesh_h.num_cells()

    upperboundary = UpperBoundary()
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
    F = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx + eps*eps*inner(u, grad(u3)) * v3 * dx + eps*eps*inner(grad(u3),grad(v3)) * dx - p * div(v) * dx + q * div(u) * dx - f1 * v1 * dx - f3 * v3 * dx - inner(theta, v) * ds(1)
    
    F = action(F, up_)
    J = derivative(F, up_)
    
    problem = NonlinearVariationalProblem(F, up_, bcu, J)
    solver  = NonlinearVariationalSolver(problem)
    solver.solve()

    (u,p) = up_.split(True)

    ufile_pvd_anis = File(resultsfolder + "velocity" + foldermarker + "/velocity_anis_degree" + str(vertical_velocity_degree) + "_eps_" + str(eps) + "_nr_cells_" + str(nr_cells) + ".pvd")
    pfile_pvd_anis = File(resultsfolder + "pressure" + foldermarker + "/pressure_anis_degree" + str(vertical_velocity_degree) + "_eps_" + str(eps) + "_nr_cells_" + str(nr_cells) + ".pvd")
    ufile_pvd_anis << u
    pfile_pvd_anis << p
    
    C = FiniteElement("Lagrange", mesh_h.ufl_cell(), degree = 1)
    C = FunctionSpace(mesh_h, C)
    bcc = concentration_BCs_pd(id, C)
    c = TrialFunction(C)
    d = TestFunction(C)
    c_sol = Function(C)
    
    # works for 0 BC concentation. otherwise, -c.dx(0)*d*ds(0)-c.dx(1)*d*ds(0) should be added, but for some reason the concentration error convergence values are not great for those cases, even if the velocity is very close
    # note that in a first order function space div(grad(u))v dx != inner(n,grad(u))ds + inner(grad(u),grad(v))dx, since by definition, first-order spaces contain linear functions, and for these the second-order derivative div(grad(u))v vanishes. so it is important to use the weak form here instead of the original second-order Laplacian of the deffusive term
    a = inner(u, grad(c)) * d * dx + c.dx(0) * d.dx(0) * dx + c.dx(1) * d.dx(1) * dx
    
    # explanation of the degree parameter: https://fenicsproject.discourse.group/t/how-to-define-source-term-function/1893, Scan_29_Nov_2019.pdf.
    # the main idea is that "degree" is a built-in parameter in this class, we do not need to "create" it. it gets defined through the call,
    # and it probably has an effect on the degree of approximation in terms of what degree is used in the \int s * phi dx integral
    # where phi is the test function, s is the source, and in the background (probably) some sort of quadrature is used to approximate this integral.
    # the degree of the quadrature is this degree, probably, or something similar.
    # Also: if degree is set to a high value, e.g. degree = 20, a warning message comes from fenics:
    # "WARNING: The number of integration points for each cell will be: 144"
    # i.e. this degree variable is responsable for the number of integration points
    anis_c_source_instance = anis_c_source(eps, id, degree = 10)
    L = inner(anis_c_source_instance, d) * dx
    
    A, b = assemble_system(a, L, bcc)
    
    solver = KrylovSolver('gmres', 'ilu')
    solver.solve(A, c_sol.vector(), b)
    
    cfile_pvd_anis = File(resultsfolder + "concentration" + foldermarker + "/concentration_anis_degree" + str(vertical_velocity_degree) + "_eps_" + str(eps) + "_nr_cells_" + str(nr_cells) + ".pvd")
    cfile_pvd_anis << c_sol
    print(eps)
    print("ANIS. c: %.15g" % c_sol.vector().norm("l2"))
    
    return [up_, c_sol]

def difference_info(eps, upc_sol_anis_eps, VPA, upc_sol_hydr, VPH):

    up_sol_anis_eps = upc_sol_anis_eps[0]
    c = upc_sol_anis_eps[1]
    (u, p) = up_sol_anis_eps.split(True)
    (u1, u3) = u.split(True)
    
    global_lists.anisotropic_norm_u1_values.append(u1.vector().norm("l2"))
    global_lists.anisotropic_norm_u3_values.append(u3.vector().norm("l2"))
    global_lists.anisotropic_norm_p_values.append(p.vector().norm("l2"))
    global_lists.anisotropic_norm_c_values.append(c.vector().norm("l2"))
    
    up_interpolate_hydr = Function(VPH)
    up_interpolate_hydr = interpolate(up_sol_anis_eps, VPH)
    (u_interpolate_hydr, p_interpolate_hydr) = up_interpolate_hydr.split(True)
    (u1_interpolate_hydr, u3_interpolate_hydr) = u_interpolate_hydr.split(True)
    
    global_lists.anisotropic_interpolated_norm_u1_values.append(u1_interpolate_hydr.vector().norm("l2"))
    global_lists.anisotropic_interpolated_norm_u3_values.append(u3_interpolate_hydr.vector().norm("l2"))
    global_lists.anisotropic_interpolated_norm_p_values.append(p_interpolate_hydr.vector().norm("l2"))
    
    up_sol_hydr = upc_sol_hydr[0]
    c_sol_hydr = upc_sol_hydr[1]
    (u_sol_hydr, p_sol_hydr) = up_sol_hydr.split(True)
    (u1_sol_hydr, u3_sol_hydr) = u_sol_hydr.split(True)
    
    global_lists.interpolated_and_hydr_difference_norm_u1_values.append((u1_interpolate_hydr.vector() - u1_sol_hydr.vector()).norm("l2"))
    global_lists.interpolated_and_hydr_difference_norm_u3_values.append((u3_interpolate_hydr.vector() - u3_sol_hydr.vector()).norm("l2"))
    global_lists.interpolated_and_hydr_difference_norm_p_values.append((p_interpolate_hydr.vector() - p_sol_hydr.vector()).norm("l2"))
    
    global_lists.anis_and_hydr_difference_norm_u1_values.append((u1.vector() - u1_sol_hydr.vector()).norm("l2"))
    # this does not make sense for different degree spaces
    #anis_and_hydr_difference_norm_u3_values.append((u3.vector() - u3_sol_hydr.vector()).norm("l2"))
    global_lists.anis_and_hydr_difference_norm_p_values.append((p.vector() - p_sol_hydr.vector()).norm("l2"))
    global_lists.anis_and_hydr_difference_norm_c_values.append((c.vector() - c_sol_hydr.vector()).norm("l2"))
    
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
    
def solve_on_refined_domain(problem_data, nx, eps, vertical_velocity_degree_anis, theta, foldermarker):
    mesh_h = UnitSquareMesh(nx, nx)
    VP = VP_functionspace(mesh_h, vertical_velocity_degree_anis)
    bcu = boundaryconditions_pd(problem_data.id, VP)
    upc_sol_anis_eps = anisotropic_solver(VP, eps, vertical_velocity_degree_anis, mesh_h, problem_data.f1, problem_data.f3, theta, bcu, foldermarker, problem_data.id)
    return upc_sol_anis_eps

def plot_error_values(problem_data):

    os.mkdir(resultsfolder + "log_errorvalues" + str(problem_data.id))
    
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_u1_problemdata_" + str(problem_data.id) + ".txt", log_errorvalues_L2_u1)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_u3_problemdata_" + str(problem_data.id) + ".txt", log_errorvalues_L2_u3)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_p_problemdata_" + str(problem_data.id) + ".txt", log_errorvalues_L2_p)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_c_problemdata_" + str(problem_data.id) + ".txt", log_errorvalues_L2_c)
    
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_u1_problemdata_" + str(problem_data.id) + ".txt", log_errorvalues_H1_u1)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_u3_problemdata_" + str(problem_data.id) + ".txt", log_errorvalues_H1_u3)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_p_problemdata_" + str(problem_data.id) + ".txt", log_errorvalues_H1_p)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_c_problemdata_" + str(problem_data.id) + ".txt", log_errorvalues_H1_c)
    
    plt.scatter(nxvalues, log_errorvalues_L2_u1)
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_u1__id" + str(problem_data.id) + ".png")
    plt.clf()
    
    plt.scatter(nxvalues, log_errorvalues_L2_u3)
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_u3__id" + str(problem_data.id) + ".png")
    plt.clf()
    
    plt.scatter(nxvalues, log_errorvalues_L2_p)
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_p__id" + str(problem_data.id) + ".png")
    plt.clf()
    
    plt.scatter(nxvalues, log_errorvalues_L2_c)
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_c__id" + str(problem_data.id) + ".png")
    plt.clf()

    plt.scatter(nxvalues, log_errorvalues_H1_u1)
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_u1__id" + str(problem_data.id) + ".png")
    plt.clf()
    
    plt.scatter(nxvalues, log_errorvalues_H1_u3)
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_u3__id" + str(problem_data.id) + ".png")
    plt.clf()
    
    plt.scatter(nxvalues, log_errorvalues_H1_p)
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_p__id" + str(problem_data.id) + ".png")
    plt.clf()
    
    plt.scatter(nxvalues, log_errorvalues_H1_c)
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_c__id" + str(problem_data.id) + ".png")
    plt.clf()

# **********************************************
# *** Define hydrostatic variational problem ***
# **********************************************

if (doHydrostatic):
    # hydrostatic model solved without initial guess for degree 1 vertical velocity space
    wind_shear_x = 10.0
    theta = Constant((wind_shear_x, 0.0))
    f1 = Expression('0', degree = 2)
    f3 = Expression('0', degree = 2)
    vertical_velocity_degree_hydr = 1
    VPH = VP_functionspace(mesh, vertical_velocity_degree_hydr)
    up_ = Function(VPH) #initial guess for the Newton solver if filled, otherwise blank and start by default
    bcu = boundaryconditions_pd(0, VPH)
    upc_sol_hydr = hydrostatic_solver(VPH, up_, vertical_velocity_degree_hydr, mesh, f1, f3, theta, bcu)

# **********************************************
# *** Define anisotropic variational problem ***
# **********************************************
if (doAnisotropicLoop):
    eps = 1.0
    vertical_velocity_degree_anis = 2
    VP = VP_functionspace(mesh, vertical_velocity_degree_anis)
    up_sol_anis_eps = Function(VP)
    wind_shear_x = 10.0
    theta = Constant((wind_shear_x, 0.0))
    f1 = Expression('0', degree = 2)
    f3 = Expression('0', degree = 2)
    bcu = boundaryconditions_pd(0, VP)
    foldermarker = "_eps_conv"

    while eps > epsilon_lower_limit:
    
        upc_sol_anis_eps = anisotropic_solver(VP, eps, vertical_velocity_degree_anis, mesh, f1, f3, theta, bcu, foldermarker, 0)
        up_sol_anis_eps = upc_sol_anis_eps[0]
        difference_info(eps, upc_sol_anis_eps, VP, upc_sol_hydr, VPH)
        eps = eps / 2.0
    
    write_plot_tools.writedifference(vertical_velocity_degree_anis, vertical_velocity_degree_hydr, resultsfolder)

# **********************************************
# *** degree 2 for the hydrostatic weak form ***
# **********************************************
if (doAnisotropicLoop and doInitGuessHydro):
    # hydrostatic model solved with initial guess for degree 2 vertical velocity space
    vertical_velocity_degree_hydr = 2
    hydrostatic_solver(VP, up_sol_anis_eps, vertical_velocity_degree_hydr, mesh, f1, f3, theta, bcu)

# **************************************************************
# *** Define anisotropic variational problem  with degree = 1 **
# **************************************************************
if (doDegree1Anisopic):
    eps = 1.0
    vertical_velocity_degree_anis = 1
    VP = VP_functionspace(mesh, vertical_velocity_degree_anis)
    wind_shear_x = 10.0
    theta = Constant((wind_shear_x, 0.0))
    f1 = Expression('0', degree = 2)
    f3 = Expression('0', degree = 2)
    bcu = boundaryconditions_pd(0, VP)
    foldermarker = "_eps_conv"

    while eps > epsilon_lower_limit:
    
        anisotropic_solver(VP, eps, vertical_velocity_degree_anis, mesh, f1, f3, theta, bcu, foldermarker, 0)
        eps = eps / 2.0

# **************************************************************
# ************************** Loop in h *************************
# **************************************************************

if (doErrorCalc):

    vertical_velocity_degree_anis = 2
    eps = 1.0
    wind_shear_x = 0.0
    theta = Constant((wind_shear_x, 0.0))
    
    for problem_data in problem_data_list:
    
        nxvalues = []
        log_errorvalues_L2_u1 = []
        log_errorvalues_L2_u3 = []
        log_errorvalues_L2_p = []
        log_errorvalues_L2_c = []
        log_errorvalues_H1_u1 = []
        log_errorvalues_H1_u3 = []
        log_errorvalues_H1_p = []
        log_errorvalues_H1_c = []
    
        errorvalues = []
        eocvalues = []
        foldermarker = "_empirical_error_calc_pd_" + str(problem_data.id)

        nx_exp = 3
        nx = 2 ** nx_exp # to control the number of cells, UnitSquareMesh(nx, nx)
        
        h_prev = 1.0
        error_prev = [1.0, 1.0, 1.0, 1.0]
        
        while nx < 2 ** 8:
    
            upc_sol_anis_eps = solve_on_refined_domain(problem_data, nx, eps, vertical_velocity_degree_anis, theta, foldermarker)
            error_next = calculate_errorvalues(problem_data, upc_sol_anis_eps, nx)
            h_next = (1/nx)*sqrt(2)
            eocvalues.append(nx)
            
            for i in [0, 1, 2, 3]:
                if error_prev[i] > DOLFIN_EPS and error_next[i] > DOLFIN_EPS:
                    eoc_i = log(error_next[i]/error_prev[i])/log(h_next/h_prev)
                else:
                    eoc_i = -1.0
                eocvalues.append(error_next[i])
                eocvalues.append(error_prev[i])
                eocvalues.append(h_next)
                eocvalues.append(h_prev)
                eocvalues.append(eoc_i)
            
            h_prev = h_next
            error_prev = error_next
            
            nx = 2 * nx
        
        np.savetxt(resultsfolder + "eocvalues_problemdata_" + str(problem_data.id) + ".txt", eocvalues)
        np.savetxt(resultsfolder + "errorvalues_problemdata_" + str(problem_data.id) + ".txt", errorvalues)
        
        plot_error_values(problem_data)

        plot_exact_solutions(resultsfolder, nx, problem_data, foldermarker)
    

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


#class PeriodicBoundary(SubDomain):
#
#    def inside(self, x, on_boundary):
#        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
#        return bool((near(x[0], 0.0) or near(x[1], 0.0)) and
#                (not ((near(x[0], 0.0) and near(x[1], 1.0)) or
#                        (near(x[0], 1.0) and near(x[1], 0.0)))) and on_boundary)
#
#    def map(self, x, y):
#        if near(x[0], 1) and near(x[1], 1):
#            y[0] = x[0] - 1.0
#            y[1] = x[1] - 1.0
#        elif near(x[0], 1):
#            y[0] = x[0] - 1.0
#            y[1] = x[1]
#        else:   # near(x[1], 1)
#            y[0] = x[0]
#            y[1] = x[1] - 1.0
#
#def VP_functionspace_periodic(mesh, v_vert_deg):
#    pbc = PeriodicBoundary()
#    V_h = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 2) #horizontal velocity
#    V_v = FiniteElement("Lagrange", mesh.ufl_cell(), degree = v_vert_deg) #vertical velocity
#    V = V_h * V_v
#    P = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 1) #pressure
#    VP = FunctionSpace(mesh, V * P, constrained_domain = pbc)
#    return VP
#
