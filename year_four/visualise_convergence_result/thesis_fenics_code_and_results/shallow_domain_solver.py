from dolfin import *
import numpy as np

from global_constants import *
import boundary_domains
import apply_bcc

def run(a_h_switch, VP, up_, eps, vertical_velocity_degree, mesh, bcu, foldermarker, problem_data):

    nr_cells = mesh.num_cells()

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    if problem_data.domaincase == "atmosphere":
        lowerboundary = boundary_domains.LowerBoundary()
        lowerboundary.mark(boundaries, 1)
    if problem_data.domaincase == "ocean":
        upperboundary = boundary_domains.UpperBoundary()
        upperboundary.mark(boundaries, 1)
    
    ds = Measure('ds')(subdomain_data = boundaries)

    up = TrialFunction(VP)
    u, p = split(up) # u,p are "trial function" type (special to FEniCS)
    u1, u3 = split(u)
    (v, q) = TestFunctions(VP)
    v1, v3 = split(v)

    (u_, p_) = up_.split(True)
    (u1_, u3_) = u_.split(True)
    
    F_base = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx + eps**2*inner(u, grad(u3)) * v3 * dx + eps**2*inner(grad(u3),grad(v3)) * dx - p * div(v) * dx + q * div(u) * dx - problem_data.f1 * v1 * dx - problem_data.f3 * v3 * dx - inner(problem_data.theta, v) * ds(1)
    
    if a_h_switch == "hydrostatic":
        F = F_base + p.dx(1) * q.dx(1) * dx
    elif a_h_switch == "anisotropic":
        F = F_base

    F = action(F, up_)
    J = derivative(F, up_, up)
    
    # nonlinear solver for the velocity and pressure
    problem = NonlinearVariationalProblem(F, up_, bcu, J)
    solver  = NonlinearVariationalSolver(problem)
    if a_h_switch == "hydrostatic":
        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1e-9
        prm['newton_solver']['relative_tolerance'] = 1e-9
        prm['newton_solver']['maximum_iterations'] = 5
    print("Solving for u, p. Problemdata " + str(problem_data.id) + " for epsilon=" + str(float(eps)) + ".")
    solver.solve()
    
    # from now on we process the data (note the usage of u,p as auxilliary variables of "function" type
    (u, p) = up_.split(True)
    (u1, u3) = u.split(True)
    
    ufile_pvd = File(resultsfolder + "velocity" + foldermarker + "/velocity__vert_velocity_degree" + str(vertical_velocity_degree) + "__eps_" + str(float(eps)) + "__nr_cells_" + str(nr_cells) + ".pvd")
    pfile_pvd = File(resultsfolder + "pressure" + foldermarker + "/pressure__vert_velocity_degree" + str(vertical_velocity_degree) + "__eps_" + str(float(eps)) + "__nr_cells_" + str(nr_cells) + ".pvd")
    ufile_pvd << u
    pfile_pvd << p
    
    C = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 1)
    C = FunctionSpace(mesh, C)
    bcc = apply_bcc.boundaryconditions_c(problem_data, C)
        
    c = TrialFunction(C)
    d = TestFunction(C)
    c_sol = Function(C)
    
    a = inner(u, grad(c)) * d * dx + c.dx(0) * d.dx(0) * dx + c.dx(1) * d.dx(1) * dx
    
    if a_h_switch == "hydrostatic":
        L = Constant(0) * d * dx
        A, b = assemble_system(a, L, bcc)
        delta = PointSource(C, Point(0.5, 0.5), 1)
        delta.apply(b)
    elif a_h_switch == "anisotropic":
        x = SpatialCoordinate(mesh)
        L = inner(problem_data.c_source(eps, x), d) * dx
        A, b = assemble_system(a, L, bcc)
    
    solver = KrylovSolver('gmres', 'ilu')
    print("Solving for c. Problemdata " + str(problem_data.id) + " for epsilon=" + str(float(eps)) + ".")
    solver.solve(A, c_sol.vector(), b)
    print("c sol norm: " + str(c_sol.vector().norm("l2")))
    
    cfile_pvd = File(resultsfolder + "concentration" + foldermarker + "/concentration__vert_velocity_degree_" + str(vertical_velocity_degree) + "__eps_" + str(float(eps)) + "__nr_cells_" + str(nr_cells) + ".pvd")
    cfile_pvd << c_sol
    
    return [up_, c_sol]
