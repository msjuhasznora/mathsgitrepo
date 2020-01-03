from dolfin import *
from math import log

import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse
import os

import problem_data_definitions as pdd
import write_plot_tools
import helper_functions
import boundary_domains
import bcc_and_source
import global_lists
import solvers

global_lists.global_lists_init()

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
   
# **********************************************
# *** Define hydrostatic variational problem ***
# **********************************************

if (doHydrostatic):
    # hydrostatic model solved without initial guess for degree 1 vertical velocity space
    vertical_velocity_degree_hydr = 1
    VPH = helper_functions.VP_functionspace(mesh, vertical_velocity_degree_hydr)
    up_ = Function(VPH) #initial guess for the Newton solver if filled, otherwise blank and start by default
    bcu = bcc_and_source.boundaryconditions_pd(pdd.problem_data0.id, VPH)
    upc_sol_hydr = solvers.hydrostatic_solver(resultsfolder, VPH, up_, vertical_velocity_degree_hydr, mesh, bcu, pdd.problem_data0)


# **********************************************
# *** Define anisotropic variational problem ***
# **********************************************

if (doAnisotropicLoop):
    eps = 1.0
    vertical_velocity_degree_anis = 2
    VP = helper_functions.VP_functionspace(mesh, vertical_velocity_degree_anis)
    up_sol_anis_eps = Function(VP)
    
    bcu = bcc_and_source.boundaryconditions_pd(pdd.problem_data0.id, VP)
    foldermarker = "_eps_conv"

    while eps > epsilon_lower_limit:
    
        upc_sol_anis_eps = solvers.anisotropic_solver(resultsfolder, VP, eps, vertical_velocity_degree_anis, mesh, bcu, foldermarker, pdd.problem_data0)
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
    solvers.hydrostatic_solver(resultsfolder, VP, up_sol_anis_eps, vertical_velocity_degree_hydr, mesh, bcu, pdd.problem_data0)


# **************************************************************
# *** Define anisotropic variational problem  with degree = 1 **
# **************************************************************

if (doDegree1Anisopic):
    eps = 1.0
    vertical_velocity_degree_anis = 1
    VP = helper_functions.VP_functionspace(mesh, vertical_velocity_degree_anis)

    bcu = bcc_and_source.boundaryconditions_pd(pdd.problem_data0.id, VP)
    foldermarker = "_eps_conv"

    while eps > epsilon_lower_limit:
    
        solvers.anisotropic_solver(resultsfolder, VP, eps, vertical_velocity_degree_anis, mesh, bcu, foldermarker, pdd.problem_data0)
        eps = eps / 2.0


# **************************************************************
# ************************** Loop in h *************************
# **************************************************************

if (doErrorCalc):

    test_problem_data_list = [pdd.problem_data1, pdd.problem_data2, pdd.problem_data3, pdd.problem_data4]

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
    
            upc_sol_anis_eps = helper_functions.solve_on_refined_domain(resultsfolder, problem_data, nx, eps, vertical_velocity_degree_anis, foldermarker)
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
