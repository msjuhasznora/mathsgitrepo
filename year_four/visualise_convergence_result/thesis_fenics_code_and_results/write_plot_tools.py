import list_container
from dolfin import *
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import os
from global_constants import *

def hydr_info(upc_sol_hydr, vertical_velocity_degree):

    up = upc_sol_hydr[0]
    c = upc_sol_hydr[1]
    (u, p) = up.split(True)
    (u1, u3) = u.split(True)
    
    hydrostatic_values = []
    hydrostatic_values.append(u1.vector().norm("l2"))
    hydrostatic_values.append(u3.vector().norm("l2"))
    hydrostatic_values.append(p.vector().norm("l2"))
    hydrostatic_values.append(c.vector().norm("l2"))
    np.savetxt(resultsfolder + "hydrostatic_values_degree_" + str(vertical_velocity_degree)+ ".txt", hydrostatic_values)
    

def writedifference(degree_anis, degree_hydr):
    np.savetxt(resultsfolder + "anisotropic_norm_u1_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.anisotropic_norm_u1_values)
    np.savetxt(resultsfolder + "anisotropic_norm_u3_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.anisotropic_norm_u3_values)
    np.savetxt(resultsfolder + "anisotropic_norm_p_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.anisotropic_norm_p_values)
    np.savetxt(resultsfolder + "anisotropic_norm_c_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.anisotropic_norm_c_values)
    np.savetxt(resultsfolder + "anisotropic_interpolated_norm_u1_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.anisotropic_interpolated_norm_u1_values)
    np.savetxt(resultsfolder + "anisotropic_interpolated_norm_u3_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.anisotropic_interpolated_norm_u3_values)
    np.savetxt(resultsfolder + "anisotropic_interpolated_norm_p_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.anisotropic_interpolated_norm_p_values)
    np.savetxt(resultsfolder + "interpolated_and_hydr_difference_norm_u1_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.interpolated_and_hydr_difference_norm_u1_values)
    np.savetxt(resultsfolder + "interpolated_and_hydr_difference_norm_u3_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.interpolated_and_hydr_difference_norm_u3_values)
    np.savetxt(resultsfolder + "interpolated_and_hydr_difference_norm_p_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.interpolated_and_hydr_difference_norm_p_values)
    np.savetxt(resultsfolder + "anis_and_hydr_difference_norm_u1_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.anis_and_hydr_difference_norm_u1_values)
    np.savetxt(resultsfolder + "anis_and_hydr_difference_norm_p_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.anis_and_hydr_difference_norm_p_values)
    np.savetxt(resultsfolder + "anis_and_hydr_difference_norm_c_values_degree_" + str(degree_anis) + "_" + str(degree_hydr) + ".txt", list_container.anis_and_hydr_difference_norm_c_values)
    
def write_test_error_values(problem_data):

    os.mkdir(resultsfolder + "log_errorvalues" + str(problem_data.id))
    os.mkdir(resultsfolder + "errorvalues" + str(problem_data.id))
    os.mkdir(resultsfolder + "eocplots" + str(problem_data.id))

    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_u1_problemdata_" + str(problem_data.id) + ".txt", list_container.log_errorvalues_L2_u1)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_u3_problemdata_" + str(problem_data.id) + ".txt", list_container.log_errorvalues_L2_u3)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_p_problemdata_" + str(problem_data.id) + ".txt", list_container.log_errorvalues_L2_p)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L2_c_problemdata_" + str(problem_data.id) + ".txt", list_container.log_errorvalues_L2_c)

    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_u1_problemdata_" + str(problem_data.id) + ".txt", list_container.log_errorvalues_H1_u1)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_u3_problemdata_" + str(problem_data.id) + ".txt", list_container.log_errorvalues_H1_u3)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_p_problemdata_" + str(problem_data.id) + ".txt", list_container.log_errorvalues_H1_p)
    np.savetxt(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_c_problemdata_" + str(problem_data.id) + ".txt", list_container.log_errorvalues_H1_c)
    
    np.savetxt(resultsfolder + "errorvalues" + str(problem_data.id) + "/H1_u1_problemdata_" + str(problem_data.id) + ".txt", list_container.errorvalues_H1_u1)
    np.savetxt(resultsfolder + "errorvalues" + str(problem_data.id) + "/H1_u3_problemdata_" + str(problem_data.id) + ".txt", list_container.errorvalues_H1_u3)
    np.savetxt(resultsfolder + "errorvalues" + str(problem_data.id) + "/H1_p_problemdata_" + str(problem_data.id) + ".txt", list_container.errorvalues_H1_p)
    np.savetxt(resultsfolder + "errorvalues" + str(problem_data.id) + "/H1_c_problemdata_" + str(problem_data.id) + ".txt", list_container.errorvalues_H1_c)
    
    np.savetxt(resultsfolder + "eocplots" + str(problem_data.id) + "/L2_u1_eoc_problemdata_" + str(problem_data.id) + ".txt", list_container.eocvalues_L2_u1)
    np.savetxt(resultsfolder + "eocplots" + str(problem_data.id) + "/L2_u3_eoc_problemdata_" + str(problem_data.id) + ".txt", list_container.eocvalues_L2_u3)
    np.savetxt(resultsfolder + "eocplots" + str(problem_data.id) + "/L2_p_eoc_problemdata_" + str(problem_data.id) + ".txt", list_container.eocvalues_L2_p)
    np.savetxt(resultsfolder + "eocplots" + str(problem_data.id) + "/L2_c_eoc_problemdata_" + str(problem_data.id) + ".txt", list_container.eocvalues_L2_c)


def plot_error_values(problem_data):

    write_test_error_values(problem_data)

    if problem_data.id == 4:
    
        plt.rc('legend', fontsize = 10)
    
        # L2 ERRORVALUES
    
        line1 = plt.scatter(list_container.nxvalues, list_container.log_errorvalues_L2_u1, color = "blue")
        line2 = plt.scatter(list_container.nxvalues, list_container.log_errorvalues_L2_u3, color = "lightgreen")
        line3 = plt.scatter(list_container.nxvalues, list_container.log_errorvalues_L2_p, color = "coral")
        line4 = plt.scatter(list_container.nxvalues, list_container.log_errorvalues_L2_c, color = "orange")
        plt.legend((line1, line2, line3, line4), ('horizontal velocity', 'vertical velocity', 'pressure', 'concentration'))
        plt.title("Testcase " + str(problem_data.id), fontsize = 14)
        plt.ylim(top = 0)
        plt.ylim(bottom = -16)
        plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 12)
        plt.ylabel("log(L2 error)", fontsize = 12)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
        plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/L_2_errors" + str(problem_data.id) + ".pdf", bbox_inches = 'tight', pad_inches = 0)
        plt.clf()
        
        # H1 ERRORVALUES
        
        line1 = plt.scatter(list_container.nxvalues, list_container.log_errorvalues_H1_u1, color = "blue")
        line2 = plt.scatter(list_container.nxvalues, list_container.log_errorvalues_H1_u3, color = "lightgreen")
        line3 = plt.scatter(list_container.nxvalues, list_container.log_errorvalues_H1_p, color = "coral")
        line4 = plt.scatter(list_container.nxvalues, list_container.log_errorvalues_H1_c, color = "orange")
        plt.legend((line1, line2, line3, line4), ('horizontal velocity', 'vertical velocity', 'pressure', 'concentration'))
        plt.title("Testcase " + str(problem_data.id), fontsize = 14)
        plt.ylim(top = 0)
        plt.ylim(bottom = -10)
        plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 12)
        plt.ylabel("log(H1 error)", fontsize = 12)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
        plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H_1_errors" + str(problem_data.id) + ".pdf", bbox_inches = 'tight', pad_inches = 0)
        plt.clf()
        
        # H1 EOC PLOTS
        
        line1, = plt.plot(list_container.eoc_nxvalues, list_container.eocvalues_L2_u1, '-o', color = "blue")
        line2, = plt.plot(list_container.eoc_nxvalues, list_container.eocvalues_L2_u3, '-o', color = "lightgreen")
        line3, = plt.plot(list_container.eoc_nxvalues, list_container.eocvalues_L2_p, '-o', color = "coral")
        line4, = plt.plot(list_container.eoc_nxvalues, list_container.eocvalues_L2_c, '-o', color = "orange")
        plt.legend([line1, line2, line3, line4], ['horizontal velocity', 'vertical velocity', 'pressure', 'concentration'])
        plt.title("EOC values for testcase " + str(problem_data.id), fontsize = 14)
        plt.ylim(top = 4)
        plt.ylim(bottom = 1)
        plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 12)
        plt.ylabel("EOC", fontsize = 12)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
        plt.savefig(resultsfolder + "eocplots" + str(problem_data.id) + "/EOC_plots" + str(problem_data.id) + ".pdf", bbox_inches = 'tight', pad_inches = 0)
        plt.clf()
        
    else :
    
        plt.rc('legend', fontsize = 18)
    
        line = plt.scatter(list_container.nxvalues, list_container.errorvalues_H1_u1, color = "blue")
        plt.legend((line,), ('horizontal velocity',))
        plt.title("Testcase " + str(problem_data.id), fontsize = 20)
        plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
        plt.ylabel("H1 error", fontsize = 20)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gcf().subplots_adjust(left=0.2)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        plt.savefig(resultsfolder + "errorvalues" + str(problem_data.id) + "/H1_u1__id" + str(problem_data.id) + ".pdf")
        plt.clf()

        line = plt.scatter(list_container.nxvalues, list_container.errorvalues_H1_u3, color = "lightgreen")
        plt.legend((line,), ('vertical velocity',))
        plt.title("Testcase " + str(problem_data.id), fontsize = 20)
        plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
        plt.ylabel("H1 error", fontsize = 20)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gcf().subplots_adjust(left=0.2)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        plt.savefig(resultsfolder + "errorvalues" + str(problem_data.id) + "/H1_u3__id" + str(problem_data.id) + ".pdf")
        plt.clf()

        line = plt.scatter(list_container.nxvalues, list_container.errorvalues_H1_p, color = "coral")
        plt.legend((line,), ('pressure',))
        plt.title("Testcase " + str(problem_data.id), fontsize = 20)
        plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
        plt.ylabel("H1 error", fontsize = 20)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gcf().subplots_adjust(left=0.2)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        plt.savefig(resultsfolder + "errorvalues" + str(problem_data.id) + "/H1_p__id" + str(problem_data.id) + ".pdf")
        plt.clf()
        
        if problem_data.id == 1:
        
            # H_1 error values without log
            line = plt.scatter(list_container.nxvalues, list_container.errorvalues_H1_c, color = "orange")
            plt.legend((line,), ('concentration',))
            plt.title("Testcase " + str(problem_data.id), fontsize = 20)
            plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
            plt.ylabel("H1 error", fontsize = 20)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.gcf().subplots_adjust(left=0.2)
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            plt.savefig(resultsfolder + "errorvalues" + str(problem_data.id) + "/H1_c__id" + str(problem_data.id) + ".pdf")
            plt.clf()
        
        else:

            # log(H_1 error values)
            line = plt.scatter(list_container.nxvalues, list_container.log_errorvalues_H1_c, color = "orange")
            plt.legend((line,), ('concentration',))
            plt.title("Testcase " + str(problem_data.id), fontsize = 20)
            plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
            plt.ylabel("log(H1 error)", fontsize = 20)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.gcf().subplots_adjust(left=0.2)
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
            plt.savefig(resultsfolder + "log_errorvalues" + str(problem_data.id) + "/H1_c__id" + str(problem_data.id) + ".pdf")
            plt.clf()
            
            plt.rc('legend', fontsize = 16)
            
            # EOC values for c
            line, = plt.plot(list_container.eoc_nxvalues, list_container.eocvalues_L2_c, '-o', color = "orange")
            plt.legend([line,], ['concentration',])
            plt.title("EOC values for testcase " + str(problem_data.id), fontsize = 20)
            plt.xlabel("The nx parameter in mesh(nx, nx)", fontsize = 20)
            plt.ylabel("EOC", fontsize = 20)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            plt.ylim(top = 2.5)
            plt.ylim(bottom = 1.5)
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.gcf().subplots_adjust(left=0.2)
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
            plt.savefig(resultsfolder + "eocplots" + str(problem_data.id) + "/EOC_plots" + str(problem_data.id) + ".pdf")
            plt.clf()
    

def difference_info(eps, upc_sol_anis_eps, VPA, upc_sol_hydr, VPH, verbose):

   up_sol_anis_eps = upc_sol_anis_eps[0]
   c = upc_sol_anis_eps[1]
   (u, p) = up_sol_anis_eps.split(True)
   (u1, u3) = u.split(True)
   
   list_container.anisotropic_norm_u1_values.append(u1.vector().norm("l2"))
   list_container.anisotropic_norm_u3_values.append(u3.vector().norm("l2"))
   list_container.anisotropic_norm_p_values.append(p.vector().norm("l2"))
   list_container.anisotropic_norm_c_values.append(c.vector().norm("l2"))
   
   up_interpolate_hydr = Function(VPH)
   up_interpolate_hydr = interpolate(up_sol_anis_eps, VPH)
   (u_interpolate_hydr, p_interpolate_hydr) = up_interpolate_hydr.split(True)
   (u1_interpolate_hydr, u3_interpolate_hydr) = u_interpolate_hydr.split(True)
   
   list_container.anisotropic_interpolated_norm_u1_values.append(u1_interpolate_hydr.vector().norm("l2"))
   list_container.anisotropic_interpolated_norm_u3_values.append(u3_interpolate_hydr.vector().norm("l2"))
   list_container.anisotropic_interpolated_norm_p_values.append(p_interpolate_hydr.vector().norm("l2"))
   
   up_sol_hydr = upc_sol_hydr[0]
   c_sol_hydr = upc_sol_hydr[1]
   (u_sol_hydr, p_sol_hydr) = up_sol_hydr.split(True)
   (u1_sol_hydr, u3_sol_hydr) = u_sol_hydr.split(True)
   
   list_container.interpolated_and_hydr_difference_norm_u1_values.append((u1_interpolate_hydr.vector() - u1_sol_hydr.vector()).norm("l2"))
   list_container.interpolated_and_hydr_difference_norm_u3_values.append((u3_interpolate_hydr.vector() - u3_sol_hydr.vector()).norm("l2"))
   list_container.interpolated_and_hydr_difference_norm_p_values.append((p_interpolate_hydr.vector() - p_sol_hydr.vector()).norm("l2"))
   
   list_container.anis_and_hydr_difference_norm_u1_values.append((u1.vector() - u1_sol_hydr.vector()).norm("l2"))
   # this does not make sense for different degree spaces
   #anis_and_hydr_difference_norm_u3_values.append((u3.vector() - u3_sol_hydr.vector()).norm("l2"))
   list_container.anis_and_hydr_difference_norm_p_values.append((p.vector() - p_sol_hydr.vector()).norm("l2"))
   list_container.anis_and_hydr_difference_norm_c_values.append((c.vector() - c_sol_hydr.vector()).norm("l2"))
   
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

