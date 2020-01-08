from dolfin import *
from math import log

import global_lists
import bcc_and_source
import solvers

def solve_on_refined_domain(resultsfolder, problem_data, nx, eps, vertical_velocity_degree_anis, foldermarker):
    mesh_h = UnitSquareMesh(nx, nx)
    VP = VP_functionspace(mesh_h, vertical_velocity_degree_anis)
    bcu = bcc_and_source.boundaryconditions_u_p(problem_data, VP)
    upc_sol_anis_eps = solvers.anisotropic_solver(resultsfolder, VP, eps, vertical_velocity_degree_anis, mesh_h, bcu, foldermarker, problem_data)
    return upc_sol_anis_eps

# create a functionspace ((V_h, V_v), P) with given degree of V_v
def VP_functionspace(mesh, v_vert_deg):
    V_h = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 2) #horizontal velocity
    V_v = FiniteElement("Lagrange", mesh.ufl_cell(), degree = v_vert_deg) #vertical velocity
    V = V_h * V_v
    P = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 1) #pressure
    VP = FunctionSpace(mesh, V * P)
    return VP

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
    global_lists.errorvalues.append(nx)
    global_lists.errorvalues.append(Eu1)
    global_lists.errorvalues.append(Eu3)
    global_lists.errorvalues.append(Ep)
    global_lists.errorvalues.append(Ec)
    global_lists.errorvalues.append(Eu1_H)
    global_lists.errorvalues.append(Eu3_H)
    global_lists.errorvalues.append(Ep_H)
    global_lists.errorvalues.append(Ec_H)

    global_lists.nxvalues.append(nx)
    global_lists.log_errorvalues_L2_u1.append(log(max(Eu1, DOLFIN_EPS)))
    global_lists.log_errorvalues_L2_u3.append(log(max(Eu3, DOLFIN_EPS)))
    global_lists.log_errorvalues_L2_p.append(log(max(Ep, DOLFIN_EPS)))
    global_lists.log_errorvalues_L2_c.append(log(max(Ec, DOLFIN_EPS)))
    global_lists.log_errorvalues_H1_u1.append(log(max(Eu1_H, DOLFIN_EPS)))
    global_lists.log_errorvalues_H1_u3.append(log(max(Eu3_H, DOLFIN_EPS)))
    global_lists.log_errorvalues_H1_p.append(log(max(Ep_H, DOLFIN_EPS)))
    global_lists.log_errorvalues_H1_c.append(log(max(Ec_H, DOLFIN_EPS)))
    global_lists.errorvalues_H1_u1.append(Eu1_H)
    global_lists.errorvalues_H1_u3.append(Eu3_H)
    global_lists.errorvalues_H1_p.append(Ep_H)
    global_lists.errorvalues_H1_c.append(Ec_H)

    error_L2 = [Eu1, Eu3, Ep, Ec]

    return error_L2
