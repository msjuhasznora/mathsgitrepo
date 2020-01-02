import global_lists
import numpy as np

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
