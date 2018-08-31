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
from math import *
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
dispMehtod = False
# Geometry
L = 200. # length
H = 200. # height
hsize = 1.5

# Material constants
E = 6.e3 # Young modulus
nu = 0.34 # Poisson ratio

#K=1.5e3
#G= 1.5e3

#E=(9*K*G)/(3*K+G)
#nu=(3*K-2*G)/(2*(3*K+G))

PlaneStress= False

# Stopping criteria for the alternate minimization
max_iterations = 50
tolerance = 1.0e-5

# Loading
body_force = Constant((0.,0.))  # bulk load

#=======================================================================================
# Geometry and mesh generation
#=======================================================================================
meshname="fracking_hsize%g" % (hsize)

# Generate a XDMF/HDF5 based mesh from a Gmsh string
geofile = \
		"""
			lc = DefineNumber[ %g, Name "Parameters/lc" ];
			H = 200.;
			L = 200.;

			r=10.;
			a=10.;

		        Point(1) = {0, 0, 0, 5*lc};
		        Point(2) = {L, 0, 0, 5*lc};
		        Point(3) = {L, H, 0, 5*lc};
		        Point(4) = {0, H, 0, 5*lc};

		        Point(5) = {L/2, H/2+r+a, 0, 0.1*lc};
		        Point(6) = {L/2, H/2+r, 0, 0.1*lc};

		        Point(7) = {L/2, H/2-r, 0, 0.1*lc};
		        Point(8) = {L/2, H/2-r-a, 0, 0.1*lc};

		        Point(9) = {L/2, H/2, 0, 0.1*lc};



		        Line(1) = {1, 2};
		        Line(2) = {2, 3};
		        Line(3) = {3, 4};
		        Line(4) = {4, 1};
		        Line Loop(5) = {1, 2, 3, 4};


			Circle(8) = {6, 9, 7};
			Circle(9) = {7, 9, 6};
			Line Loop(11) = {8, 9};
			
			Plane Surface(30) = {5,11};


			Line(6) = {5, 6};
		        Line{6} In Surface{30};

			Line(7) = {7, 8};
		        Line{7} In Surface{30};



			Physical Surface(1) = {30};

			Physical Line(101) = {6};
			Physical Line(102) = {7};


"""%(hsize)


subdir = "meshes/"
_mesh  = Mesh() #creat empty mesh object
if not os.path.isfile(subdir + meshname + ".xdmf"):
        if MPI.rank(mpi_comm_world()) == 0:
            # Create temporary .geo file defining the mesh
            if os.path.isdir(subdir) == False:
                os.mkdir(subdir)
            fgeo = open(subdir + meshname + ".geo", "w")
            fgeo.writelines(geofile)
            fgeo.close()
            # Calling gmsh and dolfin-convert to generate the .xml mesh (as well as a MeshFunction file)
            try:
                subprocess.call(["gmsh", "-2", "-o", subdir + meshname + ".msh", subdir + meshname + ".geo"])
            except OSError:
                print("-----------------------------------------------------------------------------")
                print(" Error: unable to generate the mesh using gmsh")
                print(" Make sure that you have gmsh installed and have added it to your system PATH")
                print("-----------------------------------------------------------------------------")


            meshconvert.convert2xml(subdir + meshname + ".msh", subdir + meshname + ".xml", "gmsh")

        # Convert to XDMF
        MPI.barrier(mpi_comm_world())
        mesh = Mesh(subdir + meshname + ".xml")
        XDMF = XDMFFile(mpi_comm_world(), subdir + meshname + ".xdmf")
        XDMF.write(mesh)
        XDMF.read(_mesh)

        if os.path.isfile(subdir + meshname + "_physical_region.xml") and os.path.isfile(subdir + meshname + "_facet_region.xml"):
            if MPI.rank(mpi_comm_world()) == 0:
                mesh = Mesh(subdir + meshname + ".xml")
                subdomains = MeshFunction("size_t", mesh, subdir + meshname + "_physical_region.xml")
                boundaries = MeshFunction("size_t", mesh, subdir + meshname + "_facet_region.xml")
                HDF5 = HDF5File(mesh.mpi_comm(), subdir + meshname + "_physical_facet.h5", "w")
                HDF5.write(mesh, "/mesh")
                HDF5.write(subdomains, "/subdomains")
                HDF5.write(boundaries, "/boundaries")
                print("Finish writting physical_facet to HDF5")

        if MPI.rank(mpi_comm_world()) == 0:
            # Keep only the .xdmf mesh
            #os.remove(subdir + meshname + ".geo")
            #os.remove(subdir + meshname + ".msh")
            #os.remove(subdir + meshname + ".xml")

            # Info
            print("Mesh completed")

    # Read the mesh if existing
else:
        XDMF = XDMFFile(mpi_comm_world(), subdir + meshname + ".xdmf")
        XDMF.read(_mesh)


mesh = Mesh('meshes/fracking_hsize'+str(float(hsize))+'.xml')
mesh_fun = MeshFunction("size_t", mesh,"meshes/fracking_hsize"+str(float(hsize))+"_facet_region.xml")


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
if  dispMehtod:  
	Mehtod= 'StressBC';
else: 
	Mehtod = 'DispBC'

prefix = "L%s-H%.2f-%s-%g"%(L,H,Mehtod,hsize)
save_dir = "Elasticity_result/" + prefix + "/"

prefix_stress = "L%s-H%.2f-%s-%g"%(L,H,'DispBC',hsize)
save_dir_stress = "Elasticity_result/" + prefix_stress + "/"
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

class point_left_bottom_fixed(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[0]) * 0.01, 0.) and near((x[1]) * 0.01, 0.)

class point_right_bottom_fixed(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[0]-L) * 0.01, 0.) and near((x[1]) * 0.01, 0.)

    
e1 =  [Constant([1.,0.]),Constant((1.,0.,0.))][ndim-2]
# Initialize sub-domain instances
right=Right()
left=Left()
bottom=Bottom()
top= Top()
point_LB=point_left_bottom_fixed()
point_RB=point_right_bottom_fixed()

# define meshfunction to identify boundaries by numbers
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(9999)
right.mark(boundaries, 1) # mark top as 1
left.mark(boundaries, 2) # mark top as 2
bottom.mark(boundaries, 3) # mark top as 2
top.mark(boundaries, 4) # mark top as 2
point_LB.mark(boundaries, 5) # mark top as 2
point_RB.mark(boundaries, 6) # mark top as 2
##############################################################################################################
if  dispMehtod: 
	# To read the solution for u_ 
	u_imported = Function(V_u)
	input_file_u = HDF5File(mesh.mpi_comm(), save_dir_stress+"u_4_bc.h5", "r")
	input_file_u.read(u_imported, "solution")
	input_file_u.close()

class MyExpr(Expression):
	    def eval(self, values, x):
		point = (x[0],x[1])
		values[0]= u_imported(point)[0] 
		values[1]= u_imported(point)[1] 

	    def value_shape(self):
		return (2,)

my_expr = MyExpr(degree=1)



##############################################################################################################
sigma_R =  Constant([0.,0.]) #traction on right boundary
sigma_L =  Constant([0.,0.]) #traction on right boundary
sigma_B =  Constant([0., 0.]) #traction on right boundary
sigma_T =  Constant([0.,0.]) #traction on right boundary

##############################################################################################################
Gamma_u_0 = DirichletBC(V_u, my_expr, boundaries, 1)
Gamma_u_1 = DirichletBC(V_u, my_expr, boundaries, 2)
Gamma_u_2 = DirichletBC(V_u, my_expr, boundaries, 3)
Gamma_u_3 = DirichletBC(V_u, my_expr, boundaries, 4)



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

############################################################################################

# Dump solution to file in VTK format
File(save_dir+'/u.pvd') << u_

############################################################################################
if  not dispMehtod:  
	output_file_u = HDF5File(mpi_comm_world(), save_dir+"u_4_bc.h5", "w") 
	output_file_u.write(u_, "solution")
	output_file_u.close()







