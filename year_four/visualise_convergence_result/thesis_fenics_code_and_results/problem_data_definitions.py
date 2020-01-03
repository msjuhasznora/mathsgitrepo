from dolfin import *

class ProblemData(UserExpression):
    def __init__(self, id, u1_exact, u3_exact, p_exact, f1, f3, c_exact, theta):
        self.id = id
        self.u1_exact = u1_exact
        self.u3_exact = u3_exact
        self.p_exact = p_exact
        self.f1 = f1
        self.f3 = f3
        self.c_exact = c_exact
        self.theta = theta

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
problem_data0 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact, theta)

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
problem_data1 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact, theta)

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
problem_data2 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact, theta)

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
problem_data3 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact, theta)

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
problem_data4 = ProblemData(id, u1_exact, u3_exact, p_exact, f1, f3, c_exact, theta)
