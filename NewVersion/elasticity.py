# FEnics code  Variational Fracture Mechanics
#
# A static solution of the variational fracture mechanics problems using the regularization AT2/AT1
# authors:
# corrado.maurini@upmc.fr
# Mostafa Mollaali
# Vahid



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
from CrkOpn_MosVah import Opening
import petsc4py
petsc4py.init()
from petsc4py import PETSc


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

# Parameters of the nonlinear solver used for the damage problem
snes_solver_parameters_bounds = {
 'nonlinear_solver': 'snes',
 'snes_solver': {'absolute_tolerance': 1.0e-5,
                 'line_search': 'basic',
                 'linear_solver': 'mumps',
                 'lu_solver': {'reuse_factorization': False},
                 'maximum_iterations': 50,
                 'method': 'vinewtonrsls',
                 'relative_tolerance': 1.0e-5,
                 'report': False},
 'symmetric': True}


#=======================================================================================
# Input date
#=======================================================================================
# Geometry
L = 1.0 # length
H = 0.4 # height
hsize= 0.01 # target cell size

# Material constants
E = 10. # Young modulus
nu = 0.3 # Poisson ratio

PlaneStress= False

# Stopping criteria for the alternate minimization
max_iterations = 50
tolerance = 1.0e-5

# Loading
body_force = Constant((0.,0.))  # bulk load

#=======================================================================================
# Geometry and mesh generation
#=======================================================================================
mesh = RectangleMesh(Point(0., 0.), Point(1., 0.4), 100, 40)
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
ell=0.1
law='elasticity'
prefix = "L%s-H%.2f-S%.4f"%(L,H,hsize)
save_dir = "Elasticity_result/" + prefix + "/"

if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)

# zero and unit vectors
zero_v = Constant((0.,)*ndim)
e1 = [Constant([1.,0.]),Constant((1.,0.,0.))][ndim-2]

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
def left_boundary(x, on_boundary):
    tol = 1E-14 # tolerance for coordinate comparisons
    return on_boundary and abs(x[0]) < tol

def right_boundary(x, on_boundary):
    tol = 1E-14 # tolerance for coordinate comparisons
    return on_boundary and abs(x[0] -L ) < tol
    
e1 =  [Constant([1.,0.]),Constant((1.,0.,0.))][ndim-2]
bc_left_u = DirichletBC(V_u, zero_v, left_boundary)
bc_right_u = DirichletBC(V_u.sub(0), .1, right_boundary)
bc_u = [bc_left_u, bc_right_u]

#====================================================================================
# Define  problem and solvers
#====================================================================================
elastic_energy = psi_0(u_)*dx
external_work = dot(body_force, u_)*dx
total_energy = elastic_energy - external_work

# Residual and Jacobian of elasticity problem
F_u = derivative(total_energy, u_, u_t)
J_e = derivative(F_u, u_, u)

# Variational problem for the displacement
problem_u = LinearVariationalProblem(lhs(J_e), rhs(F_u), u_, bc_u)

# Set up the solvers                                        
solver_u = LinearVariationalSolver(problem_u)

# solve elastic problem
solver_u.solve()


# Energy calculation
energy_value=assemble(total_energy)
if mpi_process_rank == 0:
    print "The elastic energy is %g" % energy_value
    print "-----------------------------------------"

# Dump solution to file in VTK format
File(save_dir+'/u.pvd') << u_

# Plot solution
plot(u_,mode = "displacement", title = 'Displacement')
plot(sqrt(inner(sigma0(u_),sigma0(u_))), title = 'stress norm')

