from dolfin import *

import boundary_domains

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


class anis_c_source(UserExpression):

    def __init__(self,eps,id,**kwargs):
        # Call superclass constructor with keyword arguments to properly
        # set up the instance:
        super().__init__(**kwargs)
        # Perform custom setup tasks for the subclass after that:
        self.eps = eps
        self.id = id

    def eval(self, values, x):
        eps = self.eps
        id = self.id
        
        if id == 0:
            # id = 0, original case
            # https://en.wikipedia.org/wiki/Cauchy_distribution#Multivariate_Cauchy_distribution
            # An example of a bivariate Cauchy distribution can be given by:
            values[0] = (1/(2 * pi)) * (eps / ( (x[0] - 0.5)**2 + (x[1] - 0.5)**2 + eps**2 )**(1.5) )
        
        elif id == 1:
            values[0] = 0
        
        elif id == 2:
            values[0] = 2*x[0]*(1-x[0]) + 2*x[1]*(1-x[1])
        
        elif id == 3:
            values[0] = 2*x[0]*(-x[0] + 1) + x[0]*(-x[0]*x[1]*(-x[1] + 1) + x[1]*(-x[0] + 1)*(-x[1] + 1)) + 2*x[1]*(-x[1] + 1) - x[1]*(-x[0]*x[1]*(-x[0] + 1) + x[0]*(-x[0] + 1)*(-x[1] + 1))
        
        elif id == 4:
            # test case 4
            values[0] = 8*pi**2*(-x[0] + 1)*(-x[1] + 1)*sin(2*pi*x[0])*sin(2*pi*x[1]) + 4*pi*(-x[0] + 1)*sin(2*pi*x[0])*cos(2*pi*x[1]) + 4*pi*(-x[1] + 1)*sin(2*pi*x[1])*cos(2*pi*x[0]) - (2*pi*(-x[0] + 1)*(-x[1] + 1)*sin(2*pi*x[0])*cos(2*pi*x[1]) - (-x[0] + 1)*sin(2*pi*x[0])*sin(2*pi*x[1]))*sin(2*pi*x[1])*cos(2*pi*x[0]) + (2*pi*(-x[0] + 1)*(-x[1] + 1)*sin(2*pi*x[1])*cos(2*pi*x[0]) - (-x[1] + 1)*sin(2*pi*x[0])*sin(2*pi*x[1]))*sin(2*pi*x[0])*cos(2*pi*x[1])
        
        else:
            values[0] = 0

    def value_shape(self):
        return ()
