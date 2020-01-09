from dolfin import *

class LateralBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and (near(x[0], 0.0, tol) or near(x[0], 1.0, tol))

class UpperBottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and (near(x[1], 0.0, tol) or near(x[1], 1.0, tol))

class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[1], 1.0, tol)
    
class LowerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[1], 0.0, tol)
        
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], 0.0, tol)
 
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], 1.0, tol)

class UnderwaterBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and not near(x[1], 1.0, tol)
