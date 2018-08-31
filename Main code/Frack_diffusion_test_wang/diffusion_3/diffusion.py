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

#=======================================================================================


#=======================================================================================
# Setup and parameters
#=======================================================================================
set_log_level(INFO)



#=======================================================================================
# Fracking Function
#=======================================================================================
def Fracking(hsize, mu_dynamic, pressure_steps,j):
	#=======================================================================================
	# Input date
	#=======================================================================================
	# Geometry
	L = 200.0 # length
	H = 200.0 # height
	#hsize= 0.5 # target cell size
	meshname="fracking_hsize%g" % (hsize)



	# Stopping criteria for the alternate minimization
	max_iterations = 200
	tolerance = 1.0e-5

	# Loading
	pressure_min = 0. # load multiplier min value
	pressure_max = 5. # load multiplier max value
	#pressure_steps = 20 # number of time steps


	#====================================================================================
	# To define  pressure field
	#====================================================================================
	f = Constant(0)


	kappa = 1.e-12
	M_biot = 1.
	#mu_dynamic = 1.


	nu =0.34
	E=6.e3
	G =E/(2.*(1.+nu))
	c = (2.*G*(1.-nu)/(1.-2.*nu))*kappa/mu_dynamic
	print c
	print G


	#t_stop = hsize**2 *mu_dynamic/(kappa*M_biot)  # non dimensional time (s) 
	#DeltaT=1.e-4#COE*hsize**2 *mu_dynamic/(kappa*M_biot)  
	DeltaT=1e-3#hsize**2/c
	print DeltaT

	#Q=5.e-4
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
	#mesh_fun = MeshFunction("size_t", mesh,"meshes/fracking_hsize"+str(float(hsize))+"_facet_region.xml")
	#plt.figure(1)
	#plot(mesh, "2D mesh")
	#plt.interactive(False)

	ndim = mesh.geometry().dim() # get number of space dimensions
	#=======================================================================================
	# Constitutive functions of the damage model
	#=======================================================================================
	def a(alpha_):
		"""
		Modulation of the elastic stiffness
		"""
		if law == "AT1":
		    return (1-alpha_)**2
		elif law == "AT2":
		    return (1-alpha_)**2
		elif law == "ATk":
		    return (1-w(alpha_))/(1+(k-1)*w(alpha_))

	def w(alpha_):
		"""
		Local energy dissipation
		"""
		if law == "AT1":
		    return alpha_
		elif law == "AT2":
		    return alpha_**2
		elif law == "ATk":
		    return 1-(1-alpha_)**2

	#=======================================================================================
	# strain, stress and strain energy for Isotropic and Amor's model
	#=======================================================================================
	#=======================================================================================
	# others definitions
	#=======================================================================================
	prefix = "L%s-H%.2f-S%.4f-mu%s"%(L,H,hsize, j)
	save_dir = "Fracking_result/" + prefix + "/"

	if os.path.isdir(save_dir):
	    shutil.rmtree(save_dir)

	# zero and unit vectors
	zero_v = Constant((0.,)*ndim)
	e1 = [Constant([1.,0.]),Constant((1.,0.,0.))][ndim-2]
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
	V_p = FunctionSpace(mesh, "CG", 1)

	# Define the function, test and trial fields
	P_, P, P_t= Function(V_p), TrialFunction(V_p), TestFunction(V_p),

	P_0 = interpolate(Expression("0.0", degree=1), V_p)
	#=======================================================================================
	# Dirichlet boundary condition for a traction test boundary
	#=======================================================================================
	## bc - P (imposed pressure)
	P_C = Expression("p", p=0.0, degree=1)
	Gamma_P_0 = DirichletBC(V_p, 0.0, boundaries, 1)
	Gamma_P_1 = DirichletBC(V_p, 0.0, boundaries, 2)
	Gamma_P_2 = DirichletBC(V_p, 0.0, boundaries, 3)
	Gamma_P_3 = DirichletBC(V_p, 0.0, boundaries, 4)
	Gamma_P_4 = DirichletBC(V_p, 1.0, boundaries, 5)

	bc_P = [ Gamma_P_0, Gamma_P_1, Gamma_P_2, Gamma_P_3, Gamma_P_4]
	#====================================================================================
	# Define  problem and solvers
	#====================================================================================
	Pressure = P_*P*dx +DeltaT*c*inner(nabla_grad(P), nabla_grad(P_))*dx-(P_0 +DeltaT*f)*P*dx#-DeltaT*Q*P*ds(1) # Wang et. al 2017 eq(8)

	#Jacobian of pressure problem
	J_p  = derivative(Pressure, P_, P_t) 

	problem_pressure = NonlinearVariationalProblem(Pressure, P_, bc_P, J=J_p)
	solver_pressure = NonlinearVariationalSolver(problem_pressure)     

	#=======================================================================================
	# To store results
	#=======================================================================================
	results = []
	file_p = File(save_dir+"/p.pvd") 
	#=======================================================================================
	# Solving at each timestep
	#=======================================================================================
	load_multipliers = np.linspace(pressure_min,pressure_max,pressure_steps)



	iteration = 0; err_P = 1; 
	# Iterations

	while  err_P>tolerance and iteration<max_iterations:
	     	# solve pressure problem
		solver_pressure.solve()
	       	err_P = (P_.vector() - P_0.vector()).norm('linf')
	       	if mpi_comm_world().rank == 0:
	       		print "Iteration:  %2d, pressure_Error: %2.8g, P_max: %.8g" %(iteration, err_P, P_.vector().max())

	       	iteration += 1
	    	P_0.vector()[:] = P_.vector()
		file_p << P_
	    
	#plt.figure(1)	
	#plot(P_, key = "P", title = "Pressure %.4f"%(p*pressure_max)) 
	#plt.show()   
	#plt.interactive(True)


if __name__ == '__main__':
        Fracking()
