from dolfin import *
import numpy as np

import boundary_domains
import bcc_and_source

def hydrostatic_solver(resultsfolder, VP, up_, vertical_velocity_degree, mesh_h, bcu, problem_data):

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
    
    # concentration
    C = FiniteElement("Lagrange", mesh_h.ufl_cell(), degree = 1)
    C = FunctionSpace(mesh_h, C)
    zerotop_concentration = DirichletBC(C, 0, "on_boundary && x[1] > 1 - DOLFIN_EPS")
    bcc = [zerotop_concentration]
    c = TrialFunction(C)
    d = TestFunction(C)
    c_sol = Function(C)
    
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
    
    hydrostatic_values.append(c_sol.vector().norm("l2"))
    np.savetxt(resultsfolder + "hydrostatic_values_degree_" + str(vertical_velocity_degree)+ ".txt", hydrostatic_values)
    
    return [up_, c_sol]
    
def anisotropic_solver(resultsfolder, VP, eps, vertical_velocity_degree, mesh_h, bcu, foldermarker, problem_data):

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
    bcc = bcc_and_source.concentration_BCs_pd(problem_data, C)
    c = TrialFunction(C)
    d = TestFunction(C)
    c_sol = Function(C)
    
    a = inner(u, grad(c)) * d * dx + c.dx(0) * d.dx(0) * dx + c.dx(1) * d.dx(1) * dx
    
    anis_c_source_instance = bcc_and_source.anis_c_source(eps, problem_data.id, degree = 10)
    L = inner(anis_c_source_instance, d) * dx
    
    A, b = assemble_system(a, L, bcc)
    
    solver = KrylovSolver('gmres', 'ilu')
    solver.solve(A, c_sol.vector(), b)
    
    cfile_pvd_anis = File(resultsfolder + "concentration" + foldermarker + "/concentration_anis_degree" + str(vertical_velocity_degree) + "_eps_" + str(eps) + "_nr_cells_" + str(nr_cells) + ".pvd")
    cfile_pvd_anis << c_sol
    print(eps)
    print("ANIS. c: %.15g" % c_sol.vector().norm("l2"))
    
    return [up_, c_sol]
