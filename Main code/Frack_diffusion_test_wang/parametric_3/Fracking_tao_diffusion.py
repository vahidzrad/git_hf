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
import petsc4py
petsc4py.init()
from petsc4py import PETSc

from math import hypot, atan2,erfc
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

# parameters of the nonlinear solver used for the alpha-problem
g6solver_alpha_parameters={"method", "tron", 			# when using gpcg make sure that you have a constant Hessian
			   "monitor_convergence", True,
	                   #"line_search", "gpcg"
	                   #"line_search", "nash"
	                   #"preconditioner", "ml_amg"
			   "report", True}

#=======================================================================================
# Fracking Function
#=======================================================================================
def Fracking(E, nu, hsize, ell, law, ModelB, ka, kb, k, mu_dynamic):

	#=======================================================================================
	# Input date
	#=======================================================================================
	# Geometry
	L = 200. # length
	H = 200. # height
	#hsize= 0.8 # target cell size
	radius = Constant(10.)
	meshname="fracking_hsize%g" % (hsize)

	# Material constants
	#E = 6.e3 # Young modulus
	#nu = 0.34 # Poisson ratio



	PlaneStress= False


	# Stopping criteria for the alternate minimization
	max_iterations = 200
	tolerance = 1.0e-8

	# Loading
	body_force = Constant((0.,0.))  # bulk load
	pressure_min = 0. # load multiplier min value
	pressure_max = 1. # load multiplier max value
	pressure_steps = 100 # number of time steps

	#====================================================================================
	# To define  pressure field
	#====================================================================================
	biot=1.0
	kappa= 1.e-12 #is the permeability of the rock #m^2
	#mu_dynamic= 0.79e-9  #is the dynamic viscosity of the fluid #MPa

	f = Constant(0)

	G=E/(2.*(1.+nu))
	c=(2.*G*(1.-nu)/(1.-2.*nu))*kappa/mu_dynamic

	DeltaT = 1e-3#hsize**2 *mu_dynamic/(kappa*M_biot)  # non dimensional time (s) 

	#=======================================================================================
	# Geometry and mesh generation
	#=======================================================================================
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
	prefix = "%s-L%s-H%.2f-S%.4f-l%.4f-ka%s-kb%s-steps%s,mu%s"%(law,L,H,hsize, ell, ka, kb, pressure_steps,k)
	save_dir = "Fracking_result/" + prefix + "/"

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
	# Define boundary sets for boundary conditions
	#=======================================================================================
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
		return near((x[1]) * 0.01, 0)

	class Void(SubDomain):
	    def inside(self, x, on_boundary):
		#return   x[0] <= 2.5 and x[0] >= 1.5  and  x[1] <= 2.5 and x[1] >= 1.5 and on_boundary
		H = 200.
		L = 200.
		radius = 10.
		return   x[0] <= L/2+1.5*radius and x[0] >= L/2-1.5*radius  and  x[1] <= H/2+1.5*radius and x[1] >= H/2-1.5*radius and on_boundary


	# Initialize sub-domain instances
	right=Right()
	left=Left()
	top=Top()
	bottom=Bottom()
	void=Void()



	# define meshfunction to identify boundaries by numbers
	boundaries = FacetFunction("size_t", mesh)
	boundaries.set_all(9999)
	right.mark(boundaries, 1) # mark top as 1
	left.mark(boundaries, 2) # mark top as 2
	top.mark(boundaries, 3) # mark top as 3
	bottom.mark(boundaries, 4) # mark bottom as 4
	void.mark(boundaries, 5) # mark void as 5


	# Define new measure including boundary naming 
	ds = Measure("ds")[boundaries] # left: ds(1), right: ds(2)
	#=======================================================================================
	# Variational formulation
	#=======================================================================================
	# Create function space for 2D elasticity + Damage
	V_u = VectorFunctionSpace(mesh, "CG", 1)
	V_p = FunctionSpace(mesh, "CG", 1)

	# Define the function, test and trial fields
	u_, u, u_t = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
	P_, P, P_t= Function(V_p), TrialFunction(V_p), TestFunction(V_p),


	P_0 = interpolate(Expression("0.0", degree=1), V_p)
	u_0 = interpolate(zero_v, V_u)
	##############################################################################################################
	# read the displacement data from a plain strain elasticity problem imposed by in-situ stress
	prefix_stress = "L%s-H%.2f-%s-%g"%(L,H,'DispBC',hsize)
	save_dir_stress = "Elasticity_result/" + prefix_stress + "/"

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
	#=======================================================================================
	# Dirichlet boundary condition for a traction test boundary
	#=======================================================================================
	Gamma_u_0 = DirichletBC(V_u, my_expr, boundaries, 1)
	Gamma_u_1 = DirichletBC(V_u, my_expr, boundaries, 2)
	Gamma_u_2 = DirichletBC(V_u, my_expr, boundaries, 3)
	Gamma_u_3 = DirichletBC(V_u, my_expr, boundaries, 4)
	bc_u =[Gamma_u_0, Gamma_u_1, Gamma_u_2, Gamma_u_3]

	## bc - P (imposed pressure)
	P_C = Expression("p", p=0.0, degree=1)

	Gamma_P_0 = DirichletBC(V_p, 0.0, boundaries, 1)
	Gamma_P_1 = DirichletBC(V_p, 0.0, boundaries, 2)
	Gamma_P_2 = DirichletBC(V_p, 0.0, boundaries, 3)
	Gamma_P_3 = DirichletBC(V_p, 0.0, boundaries, 4)
	#Gamma_P_4 = DirichletBC(V_p, P_C, mesh_fun, 101)
	Gamma_P_4 = DirichletBC(V_p, 1.0, boundaries, 5)
	bc_P = [Gamma_P_0, Gamma_P_1, Gamma_P_2, Gamma_P_3, Gamma_P_4]

	#====================================================================================
	# Define  problem and solvers
	#====================================================================================
	Pressure = P_*P*dx +DeltaT*c*inner(nabla_grad(P), nabla_grad(P_))*dx-(P_0 +DeltaT*f)*P*dx#-DeltaT*Q*P*ds(5) # Wang et. al 2017 eq(8)
	#------------------------------------------------------------------------------------
	elastic_energy = psi_0(u_)*dx- biot* P_ * div(u_) * dx
	external_work = dot(body_force, u_)*dx #+ dot(sigma_R, u_)*ds(1)+ dot(sigma_T, u_)*ds(3)

	total_energy = elastic_energy + external_work



	# Residual and Jacobian of elasticity problem
	Du_total_energ = derivative(total_energy, u_, u_t)
	J_u = derivative(Du_total_energ, u_, u)

	#Jacobian of pressure problem
	J_p  = derivative(Pressure, P_, P_t) 

	# Variational problem for the displacement
	problem_u = NonlinearVariationalProblem(Du_total_energ, u_, bc_u, J_u)


	# Parse (PETSc) parameters
	parameters.parse()


	# Set up the solvers                                        
	solver_u = NonlinearVariationalSolver(problem_u)   
	prm = solver_u.parameters
	prm["newton_solver"]["absolute_tolerance"] = 1E-6
	prm["newton_solver"]["relative_tolerance"] = 1E-6
	prm["newton_solver"]["maximum_iterations"] = 200
	prm["newton_solver"]["relaxation_parameter"] = 1.0
	prm["newton_solver"]["preconditioner"] = "default"
	prm["newton_solver"]["linear_solver"] = "mumps"              


	problem_pressure = NonlinearVariationalProblem(Pressure, P_, bc_P, J=J_p)
	solver_pressure = NonlinearVariationalSolver(problem_pressure)     

	#=======================================================================================
	# To store results
	#=======================================================================================
	results = []
	file_u = File(save_dir+"/u.pvd") # use .pvd if .xdmf is not working
	file_p = File(save_dir+"/p.pvd") 
	file_stress_tt = File(save_dir+"/stress_tt.pvd") 
	file_stress_rr = File(save_dir+"/stress_rr.pvd") 
	file_stress_rt = File(save_dir+"/stress_rt.pvd") 
	#=======================================================================================
	# Solving at each timestep
	#=======================================================================================


	iteration = 0; iter= 0 ; err_P = 1;  err_alpha = 1
	# Iterations

	while  iter<pressure_steps:
	     	# solve pressure problem
		solver_pressure.solve()
	       	err_P = (P_.vector() - P_0.vector()).norm('linf')
	       	if mpi_comm_world().rank == 0:
	       		print "Iteration:  %2d, pressure_Error: %2.8g, P_max: %.8g" %(iter, err_P, P_.vector().max())
		P_0.vector()[:] = P_.vector()



	     	# solve elastic problem
	       	solver_u.solve()
	       	if mpi_comm_world().rank == 0:
			print "elastic iteration: %2d" % (iter)

		u_0.vector()[:] = u_.vector()
		iter += 1


	        # Dump solution to file 
	        file_u << u_
	        file_p << P_ 

	        stress_tt = project(sigma0(u_)[0,0], V_p)

	        file_stress_tt << stress_tt

	        stress_rr = project(sigma0(u_)[1,1], V_p)
	        file_stress_rr << stress_rr

		stress_rt = project(sigma0(u_)[0,1], V_p)
		file_stress_rt << stress_rt






if __name__ == '__main__':
        Fracking()

