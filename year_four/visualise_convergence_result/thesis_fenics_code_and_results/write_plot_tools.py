import global_lists
from dolfin import *
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import os

def writedifference(degree_anis, degree_hydr, resultsfolder):
    np.savetxt(resultsfolder + "anisotropic_norm_u1_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.anisotropic_norm_u1_values)
    np.savetxt(resultsfolder + "anisotropic_norm_u3_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.anisotropic_norm_u3_values)
    np.savetxt(resultsfolder + "anisotropic_norm_p_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.anisotropic_norm_p_values)
    np.savetxt(resultsfolder + "anisotropic_norm_c_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.anisotropic_norm_c_values)
    np.savetxt(resultsfolder + "anisotropic_interpolated_norm_u1_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.anisotropic_interpolated_norm_u1_values)
    np.savetxt(resultsfolder + "anisotropic_interpolated_norm_u3_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.anisotropic_interpolated_norm_u3_values)
    np.savetxt(resultsfolder + "anisotropic_interpolated_norm_p_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.anisotropic_interpolated_norm_p_values)
    np.savetxt(resultsfolder + "interpolated_and_hydr_difference_norm_u1_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.interpolated_and_hydr_difference_norm_u1_values)
    np.savetxt(resultsfolder + "interpolated_and_hydr_difference_norm_u3_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.interpolated_and_hydr_difference_norm_u3_values)
    np.savetxt(resultsfolder + "interpolated_and_hydr_difference_norm_p_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.interpolated_and_hydr_difference_norm_p_values)
    np.savetxt(resultsfolder + "anis_and_hydr_difference_norm_u1_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.anis_and_hydr_difference_norm_u1_values)
    np.savetxt(resultsfolder + "anis_and_hydr_difference_norm_p_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.anis_and_hydr_difference_norm_p_values)
    np.savetxt(resultsfolder + "anis_and_hydr_difference_norm_c_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", global_lists.anis_and_hydr_difference_norm_c_values)


def plot_error_values(resultsfolder, problem_data):

    os.mkdir(resultsfolder + "log_errorvalues" + str(problem_data.id))
    os.mkdir(resultsfolder + "errorvalues" + str(problem_data.id))

    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_u1_problemdata_" + str(problem_data.id) + ".txt", global_lists.log_errorvalues_L2_u1)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_u3_problemdata_" + str(problem_data.id) + ".txt", global_lists.log_errorvalues_L2_u3)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_p_problemdata_" + str(problem_data.id) + ".txt", global_lists.log_errorvalues_L2_p)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_c_problemdata_" + str(problem_data.id) + ".txt", global_lists.log_errorvalues_L2_c)

    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_u1_problemdata_" + str(problem_data.id) + ".txt", global_lists.log_errorvalues_H1_u1)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_u3_problemdata_" + str(problem_data.id) + ".txt", global_lists.log_errorvalues_H1_u3)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_p_problemdata_" + str(problem_data.id) + ".txt", global_lists.log_errorvalues_H1_p)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_c_problemdata_" + str(problem_data.id) + ".txt", global_lists.log_errorvalues_H1_c)

    if problem_data.id == 1:
        colorid = "aquamarine"
    elif problem_data.id == 2:
        colorid = "coral"
    elif problem_data.id == 3:
        colorid = "lightgreen"
    elif problem_data.id == 4:
        colorid = "orange"
    else:
        colorid = "magenta"
    
    plt.scatter(global_lists.nxvalues, global_lists.log_errorvalues_L2_u1, color = colorid)
    plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
    plt.ylabel("log(L2 error)", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_u1__id" + str(problem_data.id) + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.clf()

    plt.scatter(global_lists.nxvalues, global_lists.log_errorvalues_L2_u3, color = colorid)
    plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
    plt.ylabel("log(L2 error)", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_u3__id" + str(problem_data.id) + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.clf()

    plt.scatter(global_lists.nxvalues, global_lists.log_errorvalues_L2_p, color = colorid)
    plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
    plt.ylabel("log(L2 error)", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_p__id" + str(problem_data.id) + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.clf()

    plt.scatter(global_lists.nxvalues, global_lists.log_errorvalues_L2_c, color = colorid)
    plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
    plt.ylabel("log(L2 error)", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_c__id" + str(problem_data.id) + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.clf()

    plt.scatter(global_lists.nxvalues, global_lists.log_errorvalues_H1_u1, color = colorid)
    plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
    plt.ylabel("log(H1 error)", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_u1__id" + str(problem_data.id) + ".pdf")
    plt.clf()

    plt.scatter(global_lists.nxvalues, global_lists.log_errorvalues_H1_u3, color = colorid)
    plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
    plt.ylabel("log(H1 error)", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_u3__id" + str(problem_data.id) + ".pdf")
    plt.clf()

    plt.scatter(global_lists.nxvalues, global_lists.log_errorvalues_H1_p, color = colorid)
    plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
    plt.ylabel("log(H1 error)", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_p__id" + str(problem_data.id) + ".pdf")
    plt.clf()

    plt.scatter(global_lists.nxvalues, global_lists.log_errorvalues_H1_c, color = colorid)
    plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
    plt.ylabel("log(H1 error)", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_c__id" + str(problem_data.id) + ".pdf")
    plt.clf()
    
    
    # H_1 error values without log
    
    plt.scatter(global_lists.nxvalues, global_lists.errorvalues_H1_u1, color = colorid)
    plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
    plt.ylabel("H1 error", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.savefig(resultsfolder + "errorvalues" + str(problem_data.id) + "/H1_u1__id" + str(problem_data.id) + ".pdf")
    plt.clf()

    plt.scatter(global_lists.nxvalues, global_lists.errorvalues_H1_u3, color = colorid)
    plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
    plt.ylabel("H1 error", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.savefig(resultsfolder + "errorvalues" + str(problem_data.id) + "/H1_u3__id" + str(problem_data.id) + ".pdf")
    plt.clf()

    plt.scatter(global_lists.nxvalues, global_lists.errorvalues_H1_p, color = colorid)
    plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
    plt.ylabel("H1 error", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.savefig(resultsfolder + "errorvalues" + str(problem_data.id) + "/H1_p__id" + str(problem_data.id) + ".pdf")
    plt.clf()

    plt.scatter(global_lists.nxvalues, global_lists.errorvalues_H1_c, color = colorid)
    plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
    plt.ylabel("H1 error", fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.savefig(resultsfolder + "errorvalues" + str(problem_data.id) + "/H1_c__id" + str(problem_data.id) + ".pdf")
    plt.clf()
    

def difference_info(eps, upc_sol_anis_eps, VPA, upc_sol_hydr, VPH, verbose):

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

