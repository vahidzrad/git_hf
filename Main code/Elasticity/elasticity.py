# FEnics code  Variational Fracture Mechanics
#
# A static solution of elasticity problem
# authors:
# corrado.maurini@upmc.fr
# Mostafa Mollaali
# Vahid Ziaei-Rad



from fenics import *
from dolfin import *
from mshr import *
from dolfin_utils.meshconvert import meshconvert
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy, sys, math, os, subprocess, shutil
import petsc4py
petsc4py.init()
from petsc4py import PETSc
from numpy import loadtxt 
from scipy.interpolate import interp1d


mpi_process_rank = dolfin.cpp.common.MPI_rank(mpi_comm_world())
#=======================================================================================
def vec(z):
    if isinstance(z, dolfin.cpp.Function):
        return dolfin.as_backend_type(z.vector()).vec()
    else:
       return dolfin.as_backend_type(z).vec()

def mat(A):
        return dolfin.as_backend_type(A).mat()

#=======================================================================================
# Setup and parameters
#=======================================================================================
set_log_level(INFO)

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 1
parameters["form_compiler"]["cpp_optimize"] = True

solver_u_parameters ={"linear_solver", "mumps", # prefer "superlu_dist" or "mumps" if available
			"preconditioner", "default",
			"report", False,
			"maximum_iterations", 500, #Added by Mostafa
			"relative_tolerance", 1e-5, #Added by Mostafa
			"symmetric", True,
			"nonlinear_solver", "newton"}

#=======================================================================================
# Input date
#=======================================================================================
# Geometry
L = 170 # length
H = 170 # height


# Material constants
E = 6.e3 # Young modulus
nu = 0.34 # Poisson ratio

PlaneStress= False

# Stopping criteria for the alternate minimization
max_iterations = 50
tolerance = 1.0e-5

# Loading
body_force = Constant((0.,0.))  # bulk load

#=======================================================================================
# Geometry and mesh generation
#=======================================================================================
mesh = RectangleMesh(Point(0., 0.), Point(170, 170), 100, 100)
plt.figure(1)
plot(mesh, "2D mesh")
plt.interactive(True)
ndim = mesh.geometry().dim() # get number of space dimensions
#=======================================================================================
# strain, stress and strain energy for Isotropic and Amor's model
#=======================================================================================
def eps(u_):
        """
        Geometrical strain
        """
        return sym(grad(u_))
#----------------------------------------------------------------------------------------
def sigma0(u_):
        """
        Application of the sound elasticy tensor on the strain tensor
        """
        Id = Identity(len(u_))
        return 2.0*mu*eps(u_) + lmbda*tr(eps(u_))*Id
#----------------------------------------------------------------------------------------
def psi_0(u_):
	"""
	The strain energy density for a linear isotropic ma-
	terial
	"""
	return  0.5 * lmbda * tr(eps(u_))**2 + mu * eps(u_)**2
#=======================================================================================
# others definitions
#=======================================================================================
dispMehtod = False

if not dispMehtod:  
	Mehtod= 'StressBC';
else: 
	Mehtod = 'DispBC'

prefix = "L%s-H%.2f-%s"%(L,H,Mehtod)
save_dir = "Elasticity_result/" + prefix + "/"

if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)

# zero and unit vectors
zero_v = Constant((0.,)*ndim)
e1 = [Constant([1.,0.]),Constant((1.,0.,0.))][ndim-2]
e2 = [Constant([0.,1.]),Constant((0.,1.,0.))][ndim-2]

# plane strain or plane stress
if not PlaneStress:  # plane strain
	lmbda = E*nu/((1.0+nu)*(1.0-2.0*nu))
else:  # plane stress
	lmbda = E*nu/(1.0-nu**2)

# shear modulus
mu = E / (2.0 * (1.0 + nu)) 


#=======================================================================================
# Variational formulation
#=======================================================================================
# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "CG", 1)

# Define the function, test and trial fields
u_, u, u_t = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
#=======================================================================================
# Dirichlet boundary condition for a traction test boundary
#=======================================================================================
# bc - u (imposed displacement)
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[0] - L) * 0.01, 0)
        
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[0]) * 0.01, 0.)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[1] - H) * 0.01, 0)
        
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[1]) * 0.01, 0.)

class vary(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[1]) * 0.01, 0.)

    
e1 =  [Constant([1.,0.]),Constant((1.,0.,0.))][ndim-2]
# Initialize sub-domain instances
right=Right()
left=Left()
bottom=Bottom()
top= Top()

# define meshfunction to identify boundaries by numbers
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(9999)
right.mark(boundaries, 1) # mark top as 1
left.mark(boundaries, 2) # mark top as 2
bottom.mark(boundaries, 3) # mark top as 2
top.mark(boundaries, 4) # mark top as 2


##############################################################################################################
sigma_R =  Constant([-6.,0.]) #traction on right boundary
sigma_L =  Constant([6.,0.]) #traction on right boundary
sigma_B =  Constant([0., 6.]) #traction on right boundary
sigma_T =  Constant([0.,-6.]) #traction on right boundary

##############################################################################################################
u_R = Constant([-0.0264, 0.0264]) #zero_v #Expression(("p", "0.",), p=0.0, degree=1)
u_L = Constant([0.0264, -0.0264])#zero_v
u_T = Constant([0.0264, -0.0264])#zero_v#Expression(("0.", "p",), p=0.0, degree=1)
u_B = Constant([-0.0264, 0.0264])#zero_v#Expression(("0.", "-p",), p=0.0, degree=1)

Gamma_u_0 = DirichletBC(V_u, u_R, boundaries, 1)
Gamma_u_1 = DirichletBC(V_u, u_L, boundaries, 2)
Gamma_u_2 = DirichletBC(V_u, u_B, boundaries, 3)
Gamma_u_3 = DirichletBC(V_u, u_T, boundaries, 4)

if not dispMehtod:  
	bc_u=[];
else:  
	bc_u = [Gamma_u_0, Gamma_u_1, Gamma_u_2, Gamma_u_3]

# Define new measure including boundary naming 
ds = Measure("ds")[boundaries] # left: ds(1), right: ds(2)
#====================================================================================
# Define  problem and solvers
#====================================================================================
elastic_energy = psi_0(u_)*dx

if not dispMehtod:  
	external_work = dot(body_force, u_)*dx+ dot(sigma_R, u_)*ds(1)+ dot(sigma_L, u_)*ds(2)+ dot(sigma_T, u_)*ds(4)+dot(sigma_B, u_)*ds(3);
else: 
	external_work = dot(body_force, u_)*dx

total_energy = elastic_energy - external_work

# First derivatives of energies (Residual)
Du_total_energy = derivative(total_energy, u_, u_t)

# Second derivatives of energies (Jacobian)
J_u = derivative(Du_total_energy, u_, u)

# Variational problem for the displacement
problem_u = NonlinearVariationalProblem(Du_total_energy, u_, bc_u, J_u)

# Set up the solvers  
solver_u = NonlinearVariationalSolver(problem_u)                 
#solver_u.Parameters.update(solver_u_parameters)

# solve elastic problem
solver_u.solve()

#=======================================================================================
# Some post-processing
#=======================================================================================
# Energy calculation
energy_value=assemble(total_energy)
if mpi_process_rank == 0:
    print "The elastic energy is %g" % energy_value
    print "-----------------------------------------"

u_nodal_values = u_.vector()
u_array = u_nodal_values.array()
coor = mesh.coordinates()
center = (85,85)
print u_(center)
############################################################################################

# Dump solution to file in VTK format
File(save_dir+'/u.pvd') << u_

# Plot solution
plot(u_,mode = "displacement", title = 'Displacement')
plot(sqrt(inner(sigma0(u_),sigma0(u_))), title = 'stress norm')

# Project and write stress field to post-processing file
W = TensorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
V = FunctionSpace(mesh, 'P', 1)

stress = project(sigma0(u_), V=W)
File(save_dir+"stress.pvd") << stress

s = sigma0(u_) - (1./3)*tr(sigma0(u_))*Identity(ndim)  # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))
V = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, V)
File(save_dir+"von_Mises.pvd") << von_Mises



