from dolfin import *

def boundaryconditions_c(problemdata, C):

    dirichlet_bc_c_list = []
    
    number_of_bcs = len(problemdata.bcc_list)
        
    for i in range(number_of_bcs):
    
        bcc_compact = problemdata.bcc_list[i]
        
        bcc_space = bcc_compact[0]
        bcc_where = bcc_compact[1]
        bcc_value = bcc_compact[2]
        
        if bcc_space == "c":
            new_bc_c = DirichletBC(C, bcc_value, bcc_where)
        dirichlet_bc_c_list.append(new_bc_c)

    return dirichlet_bc_c_list


def boundaryconditions_u_p(problemdata, VP):
    
    dirichlet_bc_up_list = []
        
    number_of_bcs = len(problemdata.bcu_list)
            
    for i in range(number_of_bcs):
        
        bcu_compact = problemdata.bcu_list[i]
            
        bcu_space = bcu_compact[0]
        bcu_where = bcu_compact[1]
        bcu_value = bcu_compact[2]
        
        if bcu_space == "u":
            new_bc_up = DirichletBC(VP.sub(0), bcu_value, bcu_where)
        elif bcu_space == "u1":
            new_bc_up = DirichletBC(VP.sub(0).sub(0), bcu_value, bcu_where)
        elif bcu_space == "u3":
            new_bc_up = DirichletBC(VP.sub(0).sub(1), bcu_value, bcu_where)
        elif bcu_space == "p":
            new_bc_up = DirichletBC(VP.sub(1), bcu_value, bcu_where)
            
        dirichlet_bc_up_list.append(new_bc_up)
        
    return dirichlet_bc_up_list
