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
import argparse
import numpy
from math import log
import os

import problem_data_definitions
import write_plot_tools
import global_lists
import helper_functions
import boundary_domains

global_lists.global_lists_init()

# Define constants

epsilon_lower_limit = 1.0e-07 #up 1.0e-07 c 5e-04

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

    lateral_boundary = boundary_domains.LateralBoundary()
    upper_bottom_boundary = boundary_domains.UpperBottomBoundary()
    upper_boundary = boundary_domains.UpperBoundary()
    lower_boundary = boundary_domains.LowerBoundary()
    left_boundary = boundary_domains.LeftBoundary()
    right_boundary = boundary_domains.RightBoundary()

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

test_problem_data_list = [problem_data_definitions.problem_data1, problem_data_definitions.problem_data2, problem_data_definitions.problem_data3, problem_data_definitions.problem_data4]

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

def hydrostatic_solver(VP, up_, vertical_velocity_degree, mesh_h, bcu, problem_data):

    nr_cells = mesh_h.num_cells()

    upperboundary = boundary_domains.UpperBoundary()
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
    F = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx - p * div(v) * dx + q * div(u) * dx + p.dx(1) * q.dx(1) * dx - problem_data.f1 * v1 * dx - problem_data.f3 * v3 * dx - inner(problem_data.theta, v) * ds(1)
    
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
    
def anisotropic_solver(VP, eps, vertical_velocity_degree, mesh_h, bcu, foldermarker, problem_data):

    nr_cells = mesh_h.num_cells()

    upperboundary = boundary_domains.UpperBoundary()
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
    F = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx + eps*eps*inner(u, grad(u3)) * v3 * dx + eps*eps*inner(grad(u3),grad(v3)) * dx - p * div(v) * dx + q * div(u) * dx - problem_data.f1 * v1 * dx - problem_data.f3 * v3 * dx - inner(problem_data.theta, v) * ds(1)
    
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
    bcc = concentration_BCs_pd(problem_data.id, C)
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
    anis_c_source_instance = anis_c_source(eps, problem_data.id, degree = 10)
    L = inner(anis_c_source_instance, d) * dx
    
    A, b = assemble_system(a, L, bcc)
    
    solver = KrylovSolver('gmres', 'ilu')
    solver.solve(A, c_sol.vector(), b)
    
    cfile_pvd_anis = File(resultsfolder + "concentration" + foldermarker + "/concentration_anis_degree" + str(vertical_velocity_degree) + "_eps_" + str(eps) + "_nr_cells_" + str(nr_cells) + ".pvd")
    cfile_pvd_anis << c_sol
    print(eps)
    print("ANIS. c: %.15g" % c_sol.vector().norm("l2"))
    
    return [up_, c_sol]
   
def solve_on_refined_domain(problem_data, nx, eps, vertical_velocity_degree_anis, foldermarker):
    mesh_h = UnitSquareMesh(nx, nx)
    VP = VP_functionspace(mesh_h, vertical_velocity_degree_anis)
    bcu = boundaryconditions_pd(problem_data.id, VP)
    upc_sol_anis_eps = anisotropic_solver(VP, eps, vertical_velocity_degree_anis, mesh_h, bcu, foldermarker, problem_data)
    return upc_sol_anis_eps


# **********************************************
# *** Define hydrostatic variational problem ***
# **********************************************

if (doHydrostatic):
    # hydrostatic model solved without initial guess for degree 1 vertical velocity space
    vertical_velocity_degree_hydr = 1
    VPH = VP_functionspace(mesh, vertical_velocity_degree_hydr)
    up_ = Function(VPH) #initial guess for the Newton solver if filled, otherwise blank and start by default
    bcu = boundaryconditions_pd(problem_data_definitions.problem_data0.id, VPH)
    upc_sol_hydr = hydrostatic_solver(VPH, up_, vertical_velocity_degree_hydr, mesh, bcu, problem_data_definitions.problem_data0)

# **********************************************
# *** Define anisotropic variational problem ***
# **********************************************
if (doAnisotropicLoop):
    eps = 1.0
    vertical_velocity_degree_anis = 2
    VP = VP_functionspace(mesh, vertical_velocity_degree_anis)
    up_sol_anis_eps = Function(VP)
    
    bcu = boundaryconditions_pd(problem_data_definitions.problem_data0.id, VP)
    foldermarker = "_eps_conv"

    while eps > epsilon_lower_limit:
    
        upc_sol_anis_eps = anisotropic_solver(VP, eps, vertical_velocity_degree_anis, mesh, bcu, foldermarker, problem_data_definitions.problem_data0)
        up_sol_anis_eps = upc_sol_anis_eps[0]
        write_plot_tools.difference_info(eps, upc_sol_anis_eps, VP, upc_sol_hydr, VPH, verbose)
        eps = eps / 2.0
    
    write_plot_tools.writedifference(vertical_velocity_degree_anis, vertical_velocity_degree_hydr, resultsfolder)

# **********************************************
# *** degree 2 for the hydrostatic weak form ***
# **********************************************
if (doAnisotropicLoop and doInitGuessHydro):
    # hydrostatic model solved with initial guess for degree 2 vertical velocity space
    vertical_velocity_degree_hydr = 2
    hydrostatic_solver(VP, up_sol_anis_eps, vertical_velocity_degree_hydr, mesh, bcu, problem_data_definitions.problem_data0)

# **************************************************************
# *** Define anisotropic variational problem  with degree = 1 **
# **************************************************************
if (doDegree1Anisopic):
    eps = 1.0
    vertical_velocity_degree_anis = 1
    VP = VP_functionspace(mesh, vertical_velocity_degree_anis)

    bcu = boundaryconditions_pd(problem_data_definitions.problem_data0.id, VP)
    foldermarker = "_eps_conv"

    while eps > epsilon_lower_limit:
    
        anisotropic_solver(VP, eps, vertical_velocity_degree_anis, mesh, bcu, foldermarker, problem_data_definitions.problem_data0)
        eps = eps / 2.0

# **************************************************************
# ************************** Loop in h *************************
# **************************************************************

if (doErrorCalc):

    vertical_velocity_degree_anis = 2
    eps = 1.0
    
    for problem_data in test_problem_data_list:
    
        global_lists.nxvalues = []
        global_lists.log_errorvalues_L2_u1 = []
        global_lists.log_errorvalues_L2_u3 = []
        global_lists.log_errorvalues_L2_p = []
        global_lists.log_errorvalues_L2_c = []
        global_lists.log_errorvalues_H1_u1 = []
        global_lists.log_errorvalues_H1_u3 = []
        global_lists.log_errorvalues_H1_p = []
        global_lists.log_errorvalues_H1_c = []
    
        global_lists.errorvalues = []
        global_lists.eocvalues = []
        foldermarker = "_empirical_error_calc_pd_" + str(problem_data.id)

        nx_exp = 3
        nx = 2 ** nx_exp # to control the number of cells, UnitSquareMesh(nx, nx)
        
        h_prev = 1.0
        error_prev = [1.0, 1.0, 1.0, 1.0]
        
        while nx < 2 ** 8:
    
            upc_sol_anis_eps = solve_on_refined_domain(problem_data, nx, eps, vertical_velocity_degree_anis, foldermarker)
            error_next = helper_functions.calculate_errorvalues(problem_data, upc_sol_anis_eps, nx)
            h_next = (1/nx)*sqrt(2)
            global_lists.eocvalues.append(nx)
            
            for i in [0, 1, 2, 3]:
                if error_prev[i] > DOLFIN_EPS and error_next[i] > DOLFIN_EPS:
                    eoc_i = log(error_next[i]/error_prev[i])/log(h_next/h_prev)
                else:
                    eoc_i = -1.0
                global_lists.eocvalues.append(error_next[i])
                global_lists.eocvalues.append(error_prev[i])
                global_lists.eocvalues.append(h_next)
                global_lists.eocvalues.append(h_prev)
                global_lists.eocvalues.append(eoc_i)
            
            h_prev = h_next
            error_prev = error_next
            
            nx = 2 * nx
        
        np.savetxt(resultsfolder + "eocvalues_problemdata_" + str(problem_data.id) + ".txt", global_lists.eocvalues)
        np.savetxt(resultsfolder + "errorvalues_problemdata_" + str(problem_data.id) + ".txt", global_lists.errorvalues)
        
        write_plot_tools.plot_error_values(resultsfolder, problem_data)
        helper_functions.plot_exact_solutions(resultsfolder, nx, problem_data, foldermarker)
    

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
