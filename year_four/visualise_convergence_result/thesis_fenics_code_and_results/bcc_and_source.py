from dolfin import *

import boundary_domains

def concentration_BCs_pd(id, C):

    zerotop_concentration = DirichletBC(C, 0.0, "on_boundary && x[1] > 1 - DOLFIN_EPS")
    zerobc = DirichletBC(C, 0.0, "on_boundary")

    if id == 0:
        return [zerotop_concentration]
    elif id == 1 :
        return [zerobc]
    elif id == 2:
        return [zerobc]
    elif id == 3:
        return [zerobc]
    elif id == 4:
        return [zerobc]
    else:
        return []


def boundaryconditions_pd(id, VP):

    lateral_boundary = boundary_domains.LateralBoundary()
    upper_bottom_boundary = boundary_domains.UpperBottomBoundary()
    upper_boundary = boundary_domains.UpperBoundary()
    lower_boundary = boundary_domains.LowerBoundary()
    left_boundary = boundary_domains.LeftBoundary()
    right_boundary = boundary_domains.RightBoundary()

    if id == 0:

        noslipbasin = DirichletBC(VP.sub(0), Constant((0, 0)), "on_boundary && x[1] < 1.0 - DOLFIN_EPS")
        zerotopvertical = DirichletBC(VP.sub(0).sub(1), 0, "on_boundary && x[1] > 1.0 - DOLFIN_EPS")
        bcu = [noslipbasin, zerotopvertical]
        return bcu

    elif id == 1:

        zeroboundaryu = DirichletBC(VP.sub(0), Constant((0, 0)), "on_boundary")
        bcuerrest = [zeroboundaryu]
        return bcuerrest

    elif id == 2:

        zeroboundaryu = DirichletBC(VP.sub(0), Constant((0, 0)), "on_boundary")
        bcuerrest = [zeroboundaryu]
        return bcuerrest

    elif id == 3:

        zeroleftu1 = DirichletBC(VP.sub(0).sub(0), 0.0, left_boundary)
        onerightu1 = DirichletBC(VP.sub(0).sub(0), 1.0, right_boundary)
        zeroloweru3 = DirichletBC(VP.sub(0).sub(1), 0.0, lower_boundary)
        minusoneupperu3 = DirichletBC(VP.sub(0).sub(1), -1.0, upper_boundary)

        p_lateral = Expression('0', degree = 3)
        p_upperbottom = Expression('0', degree = 3)
        pressureBClateral = DirichletBC(VP.sub(1), p_lateral, lateral_boundary)
        pressureBCupperbottom = DirichletBC(VP.sub(1), p_upperbottom, upper_bottom_boundary)

        bcuerrest = [zeroleftu1, onerightu1, zeroloweru3, minusoneupperu3, pressureBClateral, pressureBCupperbottom]
        return bcuerrest

    elif id == 4 :

        zerolateralu1 = DirichletBC(VP.sub(0).sub(0), 0.0, lateral_boundary)
        zerotopbottomu3 = DirichletBC(VP.sub(0).sub(1), 0.0, upper_bottom_boundary)
        p_lateral = Expression('0', degree = 3)
        p_upperbottom = Expression('0', degree = 3)
        pressureBClateral = DirichletBC(VP.sub(1), p_lateral, lateral_boundary)
        pressureBCupperbottom = DirichletBC(VP.sub(1), p_upperbottom, upper_bottom_boundary)
        bcuerrest = [zerolateralu1, zerotopbottomu3, pressureBClateral, pressureBCupperbottom]
        return bcuerrest
        
    else:
        return []


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
