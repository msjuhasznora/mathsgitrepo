from dolfin import *
import boundary_domains

lateral_boundary = boundary_domains.LateralBoundary()
upper_bottom_boundary = boundary_domains.UpperBottomBoundary()
upper_boundary = boundary_domains.UpperBoundary()
lower_boundary = boundary_domains.LowerBoundary()
left_boundary = boundary_domains.LeftBoundary()
right_boundary = boundary_domains.RightBoundary()
underwater_boundary = boundary_domains.UnderwaterBoundary()

class ProblemData(UserExpression):
    def __init__(self, id, u1_exact, u3_exact, p_exact, f1, f3, c_exact, theta, bcu_list, bcc_list, f_c):
        self.id = id
        self.u1_exact = u1_exact
        self.u3_exact = u3_exact
        self.p_exact = p_exact
        self.f1 = f1
        self.f3 = f3
        self.c_exact = c_exact
        self.theta = theta
        self.bcu_list = bcu_list
        self.bcc_list = bcc_list
        self.f_c = f_c

# for the symbolic computations providing the forcing terms using the exact solutions we use sympy-1.4: Documents/sympy-1.4/examples/beginner/differentiation.py


# PROBLEM 0
# the actual one we solve
id = 0
f1 = Expression('0.0', degree = 2)
f3 = Expression('0.0', degree = 2)
wind_shear_x = 10.0
theta = Constant((wind_shear_x, 0.0))
u1_exact = "unknown"
u3_exact = "unknown"
p_exact = "unknown"
c_exact = "unknown"
bc_up_1 = ["u", underwater_boundary, Constant((0, 0))]
bc_up_2 = ["u3", upper_boundary, 0]
bcu_list = [bc_up_1, bc_up_2]
bc_c_1 = ["c", upper_boundary, 0.0]
bcc_list = [bc_c_1]

class f_c_pd0(UserExpression):

    def __init__(self,eps,**kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def eval(self, values, x):
        eps = self.eps
        values[0] = (1/(2 * pi)) * (eps / ( (x[0] - 0.5)**2 + (x[1] - 0.5)**2 + eps**2 )**(1.5) )
        
    def value_shape(self):
        return ()

problem_data0 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact, theta, bcu_list, bcc_list, f_c_pd0)

# PROBLEM 1
id = 1
f1 = Expression('0.0', degree = 5)
f3 = Expression('0.0', degree = 5)
wind_shear_x = 0.0
theta = Constant((wind_shear_x, 0.0))
u1_exact = Expression('0', degree = 5)
u3_exact = Expression('0', degree = 5)
p_exact = Expression('0', degree = 5)
c_exact = Expression('0.0', degree = 5)
bc_up_1 = ["u", "on_boundary", Constant((0, 0))]
bcu_list = [bc_up_1]
bc_c_1 = ["c", "on_boundary", 0.0]
bcc_list = [bc_c_1]

class f_c_pd1(UserExpression):

    def __init__(self,eps,**kwargs):
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 0.0
        
    def value_shape(self):
        return ()

problem_data1 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact, theta, bcu_list, bcc_list, f_c_pd1)

# PROBLEM 2
id = 2
f1 = Expression('0.0', degree = 5)
f3 = Expression('0.0', degree = 5)
wind_shear_x = 0.0
theta = Constant((wind_shear_x, 0.0))
u1_exact = Expression('0', degree = 5)
u3_exact = Expression('0', degree = 5)
p_exact = Expression('0', degree = 5)
c_exact = Expression('x[0]*(1-x[0])*x[1]*(1-x[1])', degree = 5)
bc_up_1 = ["u", "on_boundary", Constant((0, 0))]
bcu_list = [bc_up_1]
bc_c_1 = ["c", "on_boundary", 0.0]
bcc_list = [bc_c_1]

class f_c_pd2(UserExpression):

    def __init__(self,eps,**kwargs):
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 2*x[0]*(1-x[0]) + 2*x[1]*(1-x[1])
        
    def value_shape(self):
        return ()

problem_data2 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact, theta, bcu_list, bcc_list, f_c_pd2)

# PROBLEM 3
id = 3
f1 = Expression('x[0]', degree = 5)
f3 = Expression('x[1]', degree = 5)
wind_shear_x = 0.0
theta = Constant((wind_shear_x, 0.0))
u1_exact = Expression('x[0]', degree = 5)
u3_exact = Expression('-x[1]', degree = 5)
p_exact = Expression('0', degree = 5)
c_exact = Expression('x[0]*(1-x[0])*x[1]*(1-x[1])', degree = 5)
bc_up_1 = ["u1", left_boundary, 0.0]
bc_up_2 = ["u1", right_boundary, 1.0]
bc_up_3 = ["u3", lower_boundary, 0.0]
bc_up_4 = ["u3", upper_boundary, -1.0]
bc_up_5 = ["p", lateral_boundary, Expression('0', degree = 3)]
bc_up_6 = ["p", upper_bottom_boundary, Expression('0', degree = 3)]
bcu_list = [bc_up_1, bc_up_2, bc_up_3, bc_up_4, bc_up_5, bc_up_6]
bc_c_1 = ["c", "on_boundary", 0.0]
bcc_list = [bc_c_1]

class f_c_pd3(UserExpression):

    def __init__(self,eps,**kwargs):
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 2*x[0]*(-x[0] + 1) + x[0]*(-x[0]*x[1]*(-x[1] + 1) + x[1]*(-x[0] + 1)*(-x[1] + 1)) + 2*x[1]*(-x[1] + 1) - x[1]*(-x[0]*x[1]*(-x[0] + 1) + x[0]*(-x[0] + 1)*(-x[1] + 1))
        
    def value_shape(self):
        return ()

problem_data3 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact, theta, bcu_list, bcc_list, f_c_pd3)

# PROBLEM 4
id = 4
f1 = Expression('2*pi*sin(2*pi*x[0])*sin(2*pi*x[1])*sin(2*pi*x[1])*cos(2*pi*x[0]) + 2*pi*sin(2*pi*x[0])*cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[1]) + 8*pi*pi*sin(2*pi*x[0])*cos(2*pi*x[1])', degree = 5)
f3 = Expression('2*pi*sin(2*pi*x[0])*sin(2*pi*x[0])*sin(2*pi*x[1])*cos(2*pi*x[1]) + 2*pi*sin(2*pi*x[1])*cos(2*pi*x[0])*cos(2*pi*x[0])*cos(2*pi*x[1]) - 8*pi*pi*sin(2*pi*x[1])*cos(2*pi*x[0])', degree = 5)
wind_shear_x = 0.0
theta = Constant((wind_shear_x, 0.0))
u1_exact = Expression('sin(2*pi*x[0])*cos(2*pi*x[1])', degree = 5)
u3_exact = Expression('-cos(2*pi*x[0])*sin(2*pi*x[1])', degree = 5)
p_exact = Expression('0', degree = 5)
c_exact = Expression('(1-x[0])*(1-x[1])*sin(2*pi*x[1])*sin(2*pi*x[0])', degree = 5)
bc_up_1 = ["u1", lateral_boundary, 0.0]
bc_up_2 = ["u3", upper_bottom_boundary, 0.0]
bc_up_3 = ["p", lateral_boundary, Expression('0', degree = 3)]
bc_up_4 = ["p", upper_bottom_boundary, Expression('0', degree = 3)]
bcu_list = [bc_up_1, bc_up_2, bc_up_3, bc_up_4]
bc_c_1 = ["c", "on_boundary", 0.0]
bcc_list = [bc_c_1]

class f_c_pd4(UserExpression):

    def __init__(self,eps,**kwargs):
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 8*(pi**2)*(-x[0] + 1)*(-x[1] + 1)*sin(2*pi*x[0])*sin(2*pi*x[1]) + 4*pi*(-x[0] + 1)*sin(2*pi*x[0])*cos(2*pi*x[1]) + 4*pi*(-x[1] + 1)*sin(2*pi*x[1])*cos(2*pi*x[0]) - (2*pi*(-x[0] + 1)*(-x[1] + 1)*sin(2*pi*x[0])*cos(2*pi*x[1]) - (-x[0] + 1)*sin(2*pi*x[0])*sin(2*pi*x[1]))*sin(2*pi*x[1])*cos(2*pi*x[0]) + (2*pi*(-x[0] + 1)*(-x[1] + 1)*sin(2*pi*x[1])*cos(2*pi*x[0]) - (-x[1] + 1)*sin(2*pi*x[0])*sin(2*pi*x[1]))*sin(2*pi*x[0])*cos(2*pi*x[1])
        
    def value_shape(self):
        return ()

problem_data4 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact, theta, bcu_list, bcc_list, f_c_pd4)
