# *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- *** --- #

# The anisotropic model works nicely with the (2,2) degree scenario, the hydrostatic scheme works for (2,1).
# The former does not really work for (2,1) as it develops strange unnatural layers in the pressure.
# The latter does not work for (2,2) as the Newton iterations do not converge --- probably because we do not have the first derivative of u3 in the scheme and u3 is second order in that case.

# note that in a first order function space div(grad(u))v dx != inner(n,grad(u))ds + inner(grad(u),grad(v))dx, since by definition, first-order spaces contain linear functions, and for these the second-order derivative div(grad(u))v vanishes. so it is important to use the weak form here instead of the original second-order Laplacian of the deffusive term


# the hydrostatic weak formulation without an initial guess (for now) is constructed with the vertical velocity space being of degree 1 and the additional constraint p.dx(1) * q.dx(1) * dx representing that we have a hydrostatic pressure. using a lower degree for the vertical velocities for the case of the primitive equations come from the article of Danilov, Gennady, Schroter, 2002 (even though they use elementwise constant representations)

# The vertical velocity is from a quadratic space. Using a degree 1 vertical velocity space in the anisotropic case we have a strange layered unnatural pressure.


# explanation of the degree parameter: https://fenicsproject.discourse.group/t/how-to-define-source-term-function/1893, Scan_29_Nov_2019.pdf.
# the main idea is that "degree" is a built-in parameter in this class, we do not need to "create" it. it gets defined through the call,
# and it probably has an effect on the degree of approximation in terms of what degree is used in the \int s * phi dx integral
# where phi is the test function, s is the source, and in the background (probably) some sort of quadrature is used to approximate this integral.
# the degree of the quadrature is this degree, probably, or something similar.
# Also: if degree is set to a high value, e.g. degree = 20, a warning message comes from fenics:
# "WARNING: The number of integration points for each cell will be: 144"
# i.e. this degree variable is responsable for the number of integration points

# BOUNDARY comments:
# 1.) the Dirichlet part of the boundary, \Gamma: you do not add the boundary terms on \Gamma to the F form itself as something like c.dx(1)*d*dx(\Gamma). instead, you define a DirichletBC condition, and then apply it to the Problem itself.
# 2.) on parts of the boundary where we have a DirichletBC defined, the test functions go to zero, thus an added boundary integral would not change anything
# 3.) in FEniCS the test functions go to zero on and only on the boundary section for which we have DirichletBC defined
# 4.) boundary integral terms in the F form should be used exactly for those boundary sections where we do not have a DirichletBC. And among these, where we do not have a Dirichlet condition, we either have a NeumannBC, or, we risk the problem being ill-posed.
# 5.) on a boundary section where we do not have a DirichletBC: in many problems the h value of the Neumann BC is zero, and the boundary term is therefore omitted; this case is sometimes referred to as the “do-nothing boundary condition”.
# 6.) something like grad(u)*v*dx(\Gamma) in itself for Neumann BC-s should not be used. you know that grad(u) = h for a given h on \Gamma, then you add h*v*dx(\Gamma) in the form F.

# improvement ideas:

# 0) add C

# i) the scheme depends epsilon-freely on the complete u. why do the Newton iterations not converge for the case when we have degree=2 for the vertical velocity? is it that the scheme contains only u3, but not grad(u3), and it is a 2-degree space? probably not in general, as for mesh(1,1) the method actually does converge.

# ii) is it possible to make it time-dependent?

# iii) 3D in space

# iv) periodicity in the x direction would make the domain into a tube with upper wind traction and fully x-directional circulation. Having a 0 y-directional velocity would be ok in itself, but with that there is not much to visualise as the scheme then does not depend on epsilon. so I think it is better to have the classical domain in order to make a point with the visualisation.

# v) print the Jacobian, makes sense for small mesh:

#J_mat = assemble(J)
#J_array = J_mat.array()
#np.savetxt("Jacobianmatrix_h2.txt", J_array)
#detJ = numpy.linalg.det(J_array)
#print("det Jacobian h 2:")
#print(detJ)


#class PeriodicBoundary(SubDomain):
#
#    def inside(self, x, on_boundary):
#        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
#        return bool((near(x[0], 0.0) or near(x[1], 0.0)) and
#                (not ((near(x[0], 0.0) and near(x[1], 1.0)) or
#                        (near(x[0], 1.0) and near(x[1], 0.0)))) and on_boundary)
#
#    def map(self, x, y):
#        if near(x[0], 1) and near(x[1], 1):
#            y[0] = x[0] - 1.0
#            y[1] = x[1] - 1.0
#        elif near(x[0], 1):
#            y[0] = x[0] - 1.0
#            y[1] = x[1]
#        else:   # near(x[1], 1)
#            y[0] = x[0]
#            y[1] = x[1] - 1.0
#
#def VP_functionspace_periodic(mesh, v_vert_deg):
#    pbc = PeriodicBoundary()
#    V_h = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 2) #horizontal velocity
#    V_v = FiniteElement("Lagrange", mesh.ufl_cell(), degree = v_vert_deg) #vertical velocity
#    V = V_h * V_v
#    P = FiniteElement("Lagrange", mesh.ufl_cell(), degree = 1) #pressure
#    VP = FunctionSpace(mesh, V * P, constrained_domain = pbc)
#    return VP
#
