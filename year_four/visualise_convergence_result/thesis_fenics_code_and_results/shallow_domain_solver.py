from dolfin import *
import numpy as np

import boundary_domains
import apply_bcc

def run(a_h_switch, resultsfolder, VP, up_, eps, vertical_velocity_degree, mesh_h, bcu, foldermarker, problem_data):

    nr_cells = mesh_h.num_cells()

    upperboundary = boundary_domains.UpperBoundary()
    boundaries = MeshFunction("size_t", mesh_h, mesh_h.topology().dim() - 1)
    boundaries.set_all(0)
    upperboundary.mark(boundaries, 1)
    ds = Measure('ds')(subdomain_data = boundaries)

    up = TrialFunction(VP)
    u, p = split(up) # u,p are "trial function" type (special to FEniCS)
    u1, u3 = split(u)
    (v, q) = TestFunctions(VP)
    v1, v3 = split(v)

    (u_, p_) = up_.split(True)
    (u1_, u3_) = u_.split(True)
    
    if a_h_switch == "hydrostatic":
        # the hydrostatic weak formulation without an initial guess (for now) is constructed with the vertical velocity space being of degree 1 and the additional constraint p.dx(1) * q.dx(1) * dx representing that we have a hydrostatic pressure. using a lower degree for the vertical velocities for the case of the primitive equations come from the article of Danilov, Gennady, Schroter, 2002 (even though they use elementwise constant representations)
        F = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx - p * div(v) * dx + q * div(u) * dx + p.dx(1) * q.dx(1) * dx - problem_data.f1 * v1 * dx - problem_data.f3 * v3 * dx - inner(problem_data.theta, v) * ds(1)
    elif a_h_switch == "anisotropic":
        # the anisotropic weak formulation is created using the Taylor-Hood elements, the vertical velocity is from a quadratic space. Using a degree 1 vertical velocity space in the anisotropic case we have a strange layered unnatural pressure.
        F = inner(u, grad(u1)) * v1 * dx + inner(grad(u1),grad(v1)) * dx + eps*eps*inner(u, grad(u3)) * v3 * dx + eps*eps*inner(grad(u3),grad(v3)) * dx - p * div(v) * dx + q * div(u) * dx - problem_data.f1 * v1 * dx - problem_data.f3 * v3 * dx - inner(problem_data.theta, v) * ds(1)

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
    print("Solving for u, p. Problemdata " + str(problem_data.id) + " for epsilon=" + str(eps) + ".")
    solver.solve()
    
    # from now on we process the data (note the usage of u,p as auxilliary variables of "function" type
    (u, p) = up_.split(True)
    (u1, u3) = u.split(True)
    
    ufile_pvd = File(resultsfolder + "velocity" + foldermarker + "/velocity__vert_velocity_degree" + str(vertical_velocity_degree) + "__eps_" + str(eps) + "__nr_cells_" + str(nr_cells) + ".pvd")
    pfile_pvd = File(resultsfolder + "pressure" + foldermarker + "/pressure__vert_velocity_degree" + str(vertical_velocity_degree) + "__eps_" + str(eps) + "__nr_cells_" + str(nr_cells) + ".pvd")
    ufile_pvd << u
    pfile_pvd << p
    
    C = FiniteElement("Lagrange", mesh_h.ufl_cell(), degree = 1)
    C = FunctionSpace(mesh_h, C)
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
        anis_c_source_instance = problem_data.f_c(eps, degree = 10)
        L = inner(anis_c_source_instance, d) * dx
        A, b = assemble_system(a, L, bcc)
    
    solver = KrylovSolver('gmres', 'ilu')
    print("Solving for c. Problemdata " + str(problem_data.id) + " for epsilon=" + str(eps) + ".")
    solver.solve(A, c_sol.vector(), b)
    print("c sol norm: " + str(c_sol.vector().norm("l2")))
    
    cfile_pvd = File(resultsfolder + "concentration" + foldermarker + "/concentration__vert_velocity_degree_" + str(vertical_velocity_degree) + "__eps_" + str(eps) + "__nr_cells_" + str(nr_cells) + ".pvd")
    cfile_pvd << c_sol
    
    return [up_, c_sol]
