from dolfin import *

import matplotlib.pyplot as plt
from math import log
import numpy as np
import os

import problem_data_definitions as pdd
from global_constants import *
import shallow_domain_solver
import write_plot_tools
import helper_functions
import list_container
import apply_bcc

doHydrostatic = True
doAnisotropicLoop = True
doInitGuessHydro = True
doDegree1Anisopic = True
doErrorCalc = True

watertop_problemdata = pdd.problem_data0
   
# **********************************************
# *** Define hydrostatic variational problem ***
# **********************************************

if (doHydrostatic):
    # hydrostatic model solved without initial guess for degree 1 vertical velocity space
    vertical_velocity_degree_hydr = 1
    VPH = helper_functions.VP_functionspace(default_mesh, vertical_velocity_degree_hydr)
    up_ = Function(VPH) #initial guess for the Newton solver if filled, otherwise blank and start by default
    bcu = apply_bcc.boundaryconditions_u_p(pdd.problem_data0, VPH)
    eps = Constant(0.0)
    foldermarker = "_hydr"
    upc_sol_hydr = shallow_domain_solver.run("hydrostatic", VPH, up_, eps, vertical_velocity_degree_hydr, default_mesh, bcu, foldermarker, watertop_problemdata)
    write_plot_tools.hydr_info(upc_sol_hydr, vertical_velocity_degree_hydr)


# **********************************************
# *** Define anisotropic variational problem ***
# **********************************************

if (doAnisotropicLoop):
    vertical_velocity_degree_anis = 2
    VP = helper_functions.VP_functionspace(default_mesh, vertical_velocity_degree_anis)
    up_ = Function(VP)
    up_sol_anis_eps = Function(VP)
    bcu = apply_bcc.boundaryconditions_u_p(pdd.problem_data0, VP)
    foldermarker = "_eps_conv"
    up_sol_anis_eps = helper_functions.solve_epsilon_loop_plus_info(watertop_problemdata, VP, up_, vertical_velocity_degree_anis, default_mesh, bcu, foldermarker, upc_sol_hydr, VPH, vertical_velocity_degree_hydr)


# **********************************************
# *** degree 2 for the hydrostatic weak form ***
# **********************************************

if (doAnisotropicLoop and doInitGuessHydro):
    # hydrostatic model solved with initial guess for degree 2 vertical velocity space
    vertical_velocity_degree_hydr = 2
    eps = Constant(0.0)
    foldermarker = "_hydr"
    upc_sol_hydr = shallow_domain_solver.run("hydrostatic", VP, up_sol_anis_eps, eps, vertical_velocity_degree_hydr, default_mesh, bcu, foldermarker, watertop_problemdata)
    write_plot_tools.hydr_info(upc_sol_hydr, vertical_velocity_degree_hydr)


# **************************************************************
# *** Define anisotropic variational problem  with degree = 1 **
# **************************************************************

if (doDegree1Anisopic):
    vertical_velocity_degree_anis = 1
    VP = helper_functions.VP_functionspace(default_mesh, vertical_velocity_degree_anis)
    up_ = Function(VP)
    bcu = apply_bcc.boundaryconditions_u_p(pdd.problem_data0, VP)
    foldermarker = "_eps_conv"
    helper_functions.solve_epsilon_loop_basic(watertop_problemdata, VP, up_, vertical_velocity_degree_anis, default_mesh, bcu, foldermarker)


# **************************************************************
# ************************** Loop in h *************************
# **************************************************************

if (doErrorCalc):

    test_problem_data_list = [pdd.problem_data1, pdd.problem_data2, pdd.problem_data3, pdd.problem_data4]

    vertical_velocity_degree_anis = 2
    eps = 1.0
    
    for problem_data in test_problem_data_list:
        
        list_container.init_error_lists()
        
        eoclists = [list_container.eocvalues_L2_u1, list_container.eocvalues_L2_u3, list_container.eocvalues_L2_p, list_container.eocvalues_L2_c]
        foldermarker = "_empirical_error_calc_pd_" + str(problem_data.id)

        nx_exp = 3
        nx = 2 ** nx_exp # to control the number of cells, UnitSquareMesh(nx, nx)
        
        h_prev = 0.25*sqrt(2)
        error_prev = [1.0, 1.0, 1.0, 1.0]
        
        while nx < 2 ** 8:
    
            upc_sol_anis_eps = helper_functions.solve_on_refined_domain("anisotropic", problem_data, nx, eps, vertical_velocity_degree_anis, foldermarker)
            error_next = helper_functions.calculate_errorvalues(problem_data, upc_sol_anis_eps, nx)
            h_next = (1/nx)*sqrt(2)
            if nx > 2 ** nx_exp:
                list_container.eocvalues.append(nx)
                list_container.eoc_nxvalues.append(nx)
            
            for i in [0, 1, 2, 3]:
                if error_prev[i] > DOLFIN_EPS and error_next[i] > DOLFIN_EPS:
                    eoc_i = log(error_next[i]/error_prev[i])/log(h_next/h_prev)
                else:
                    eoc_i = -1.0
                    
                if nx > 2 ** 3:
                    list_container.eocvalues.append(eoc_i)
                    (eoclists[i]).append(eoc_i)
            
            h_prev = h_next
            error_prev = error_next
            
            nx = 2 * nx
        
        np.savetxt(resultsfolder + "eocvalues_problemdata_" + str(problem_data.id) + ".txt", list_container.eocvalues)
        np.savetxt(resultsfolder + "errorvalues_problemdata_" + str(problem_data.id) + ".txt", list_container.errorvalues)
        
        write_plot_tools.plot_error_values(problem_data)
        helper_functions.plot_exact_solutions(nx, problem_data, foldermarker)
