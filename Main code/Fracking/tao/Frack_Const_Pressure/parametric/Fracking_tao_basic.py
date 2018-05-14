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


# parameters of the nonlinear solver used for the displacement-problem
solver_u_parameters ={"linear_solver", "mumps", # prefer "superlu_dist" or "mumps" if available
			"preconditioner", "default",
			"report", False,
			"maximum_iterations", 500, 
			"relative_tolerance", 1e-5, 
			"symmetric", True,
			"nonlinear_solver", "newton"}

#=======================================================================================
# Fracking Function
#=======================================================================================
def Fracking(hsize, pressure_max, ell,E, nu, Model, law):
	#=======================================================================================
	# Input date
	#=======================================================================================
	# Geometry
	L = 4.0 # length
	H = 4.0 # height
	#hsize= 0.01 # target cell size
	meshname="fracking_hsize%g" % (hsize)

	# Material constants
	#ell = Constant(4 * hsize) # internal length scale
	#E = 10. # Young modulus
	#nu = 0.3 # Poisson ratio

	biot= 0. #Biot coefficient

	PlaneStress= False

	gc = 1. # fracture toughness
	k_ell = Constant(1.0e-12) # residual stiffness
	#law = "AT1"

	# effective toughness 
	if law == "AT2":  
		Gc = gc/(1+hsize/(2*ell)) #AT2
	elif  law == "AT1": 
		Gc = gc/(1+3*hsize/(8*ell)) #AT1
	else:
		Gc = gc 

	Gc=1.

	ModelB= False 
	if not ModelB:  # Model A (isotropic model)
		Model = 'Isotropic'
	else:  # Model B (Amor's model)
		Model = 'Amor'

	# Stopping criteria for the alternate minimization
	max_iterations = 50
	tolerance = 1.0e-5

	# Loading
	ut = 0.0 # reference value for the loading (imposed displacement)
	body_force = Constant((0.,0.))  # bulk load
	pressure_min = 0. # load multiplier min value
	#pressure_max = 1. # load multiplier max value
	pressure_steps = 10 # number of time steps

	WheelerApproach= False

	#=======================================================================================
	# Geometry and mesh generation
	#=======================================================================================
	# Generate a XDMF/HDF5 based mesh from a Gmsh string
	geofile = \
			"""
			lc = DefineNumber[ %g, Name "Parameters/lc" ];
			H = 4;
			L = 4;
		        Point(1) = {0, 0, 0, 10*lc};
		        Point(2) = {L, 0, 0, 10*lc};
		        Point(3) = {L, H, 0, 10*lc};
		        Point(4) = {0, H, 0, 10*lc};
		        Point(5) = {1.8, H/2, 0, 1*lc};
		        Point(6) = {2.2, H/2, 0, 1*lc};
		        Line(1) = {1, 2};
		        Line(2) = {2, 3};
		        Line(3) = {3, 4};
		        Line(4) = {4, 1};
		        Line Loop(5) = {1, 2, 3, 4};
			Plane Surface(30) = {5};

			Line(6) = {5, 6};
		        Line{6} In Surface{30};


			Physical Surface(1) = {30};

			Physical Line(101) = {6};

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
	#plt.figure(1)
	#plot(mesh, "2D mesh")
	#plt.interactive(True)

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
	def angle_bracket_plus(a):
	    return (a+abs(a))/2

	def angle_bracket_minus(a):
	    return (a-abs(a))/2
	#----------------------------------------------------------------------------------------
	def g(alpha_):
		"""
		degradation function
		"""
		return ((1-alpha_)**2+k_ell) 
	#----------------------------------------------------------------------------------------
	def eps(u_):
		"""
		Geometrical strain
		"""
		return sym(grad(u_))

	def dev_eps(u_):
		"""

		"""
		return eps(u_) - 1/3*tr(eps(u_))*Identity(ndim)
	#----------------------------------------------------------------------------------------
	def sigma0(u_):
		"""
		Application of the sound elasticy tensor on the strain tensor
		"""
		Id = Identity(len(u_))
		return 2.0*mu*eps(u_) + lmbda*tr(eps(u_))*Id


	def sigma_A(u_, alpha_):
		"""
		Stress Model A
		"""
		return g(alpha_) * sigma0(u_)


	def sigma_B(u_,alpha_):
		"""
		Stress Model B
		"""
		return  g(alpha_) * ( (lmbda+2/3*mu) * ( angle_bracket_plus( tr(eps(u_))) * Identity(ndim) )+ 2*mu*dev_eps(u_) ) + (lmbda+2/3*mu)*( angle_bracket_minus(tr(dev_eps(u_))) * Identity(ndim))
	#----------------------------------------------------------------------------------------
	def psi_0(u_):
		"""
		The strain energy density for a linear isotropic ma-
		terial
		"""
		return  0.5 * lmbda * tr(eps(u_))**2 + mu * eps(u_)**2

	def psi_A(u_, alpha_):
		"""
		The strain energy density for model A
		"""
		return g(alpha_) * psi_0(u_)

	def psi_B(u_,alpha_):
		"""
		The strain energy density for model B
		"""
		return  g(alpha_) * ( 0.5*K * ( angle_bracket_plus(tr(dev_eps(u_))**2)) + mu*dev_eps(u_)**2) + 0.5*K * ( angle_bracket_minus(tr(dev_eps(u_))**2))
	#----------------------------------------------------------------------------------------

	if not ModelB:  # Model A (isotropic model)
		psi = psi_A
		sigma= sigma_A
	else:  # Model B (Amor's model)
		psi = psi_B
		sigma=sigma_B

	#=======================================================================================
	# others definitions
	#=======================================================================================
	prefix = "%s-%s-L%s-H%.2f-S%.4f-l%.4f"%(law,Model,L,H,hsize, ell)
	save_dir = "Fracking_result/" + prefix + "/"

	if os.path.isdir(save_dir):
	    shutil.rmtree(save_dir)

	# zero and unit vectors
	zero_v = Constant((0.,)*ndim)
	e1 = [Constant([1.,0.]),Constant((1.,0.,0.))][ndim-2]

	# Normalization constant for the dissipated energy 
	# to get Griffith surface energy for ell going to zero
	z = sympy.Symbol("z", positive=True)
	c_w = float(4*sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1)))

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

	# Initialize sub-domain instances
	right=Right()
	left=Left()
	top=Top()
	bottom=Bottom()


	# define meshfunction to identify boundaries by numbers
	boundaries = FacetFunction("size_t", mesh)
	boundaries.set_all(9999)
	right.mark(boundaries, 1) # mark top as 1
	left.mark(boundaries, 2) # mark top as 2
	top.mark(boundaries, 3) # mark top as 3
	bottom.mark(boundaries, 4) # mark bottom as 4

	# Define new measure including boundary naming 
	ds = Measure("ds")[boundaries] # left: ds(1), right: ds(2)
	#=======================================================================================
	# Variational formulation
	#=======================================================================================
	# Create function space for 2D elasticity + Damage
	V_u = VectorFunctionSpace(mesh, "CG", 1)
	V_alpha = FunctionSpace(mesh, "CG", 1)

	# Define the function, test and trial fields
	u_, u, u_t = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
	alpha_, alpha, alpha_t = Function(V_alpha), TrialFunction(V_alpha), TestFunction(V_alpha)

	alpha_0 = interpolate(Constant(0.0), V_alpha) # initial (known) alpha, undamaged everywhere.
	#=======================================================================================
	# Dirichlet boundary condition for a traction test boundary
	#=======================================================================================
	# Define a time dependent pressure
	p_b = Expression("t", t=0.0, degree=1)
	Pressure = interpolate(p_b, V_alpha)

	# bc - u (imposed displacement)
	u_R = zero_v
	u_L = zero_v
	u_T = Expression(("0.", "t",), t=0.0, degree=1)
	u_B = Expression(("0.", "-t",), t=0.0, degree=1)

	Gamma_u_0 = DirichletBC(V_u, u_R, boundaries, 1)
	Gamma_u_1 = DirichletBC(V_u, u_L, boundaries, 2)
	Gamma_u_2 = DirichletBC(V_u, u_T, boundaries, 3)
	Gamma_u_3 = DirichletBC(V_u, u_B, boundaries, 4)
	bc_u = [Gamma_u_0, Gamma_u_1, Gamma_u_2, Gamma_u_3]

	# bc - alpha (zero damage)
	Gamma_alpha_0 = DirichletBC(V_alpha, 0.0, boundaries, 1)
	Gamma_alpha_1 = DirichletBC(V_alpha, 0.0, boundaries, 2)
	Gamma_alpha_2 = DirichletBC(V_alpha, 0.0, boundaries, 3)
	Gamma_alpha_3 = DirichletBC(V_alpha, 0.0, boundaries, 4)
	Gamma_alpha_4 = DirichletBC(V_alpha, 1.0, mesh_fun, 101)
	bc_alpha = [Gamma_alpha_0, Gamma_alpha_1, Gamma_alpha_2, Gamma_alpha_3,Gamma_alpha_4]
	#====================================================================================
	# Define  problem and solvers
	#====================================================================================
	elastic_energy = psi(u_, alpha_)*dx
	external_work = dot(body_force, u_)*dx

	if not WheelerApproach:  # Bourdin approach, 2012 , SPE 159154, equation (5)
		pressurized_energy = Pressure * inner(u_, grad(alpha_)) *dx
	else:  # M.F. Wheeler et al., CAMAME, 2014, eqution (4)
		pressurized_energy = -(biot-1.)*((1.-alpha_)**2 * Pressure * div(u_)) * dx+ dot((1-alpha_)**2 * grad(Pressure), u_) * dx

	dissipated_energy = Gc/float(c_w)*( w(alpha_)/ell + ell* inner( grad(alpha_), grad(alpha_)) )*dx 

	total_energy = elastic_energy + dissipated_energy + pressurized_energy - external_work



	# Residual and Jacobian of elasticity problem
	Du_total_energy = derivative(total_energy, u_, u_t)
	J_u = derivative(Du_total_energy, u_, u)

	# Residual and Jacobian of damage problem
	Dalpha_total_energy = derivative(total_energy, alpha_, alpha_t)
	J_alpha = derivative(Dalpha_total_energy, alpha_, alpha)

	# Variational problem for the displacement
	problem_u = NonlinearVariationalProblem(Du_total_energy, u_, bc_u, J_u)

	# Variational problem for the damage (non-linear to use variational inequality solvers of petsc)
	# Define the minimisation problem by using OptimisationProblem class
	class DamageProblem(OptimisationProblem):

	    def __init__(self):
		OptimisationProblem.__init__(self)

	    # Objective function
	    def f(self, x):
		alpha_.vector()[:] = x
		return assemble(total_energy)

	    # Gradient of the objective function
	    def F(self, b, x):
		alpha_.vector()[:] = x
		assemble(Dalpha_total_energy, tensor=b)

	    # Hessian of the objective function
	    def J(self, A, x):
		alpha_.vector()[:] = x
		assemble(J_alpha, tensor=A)

	# Create the PETScTAOSolver
	problem_alpha = DamageProblem()

	# Parse (PETSc) parameters
	parameters.parse()


	# Set up the solvers                                        
	solver_u = NonlinearVariationalSolver(problem_u)                 
	prm = solver_u.parameters
	prm["newton_solver"]["absolute_tolerance"] = 1E-8
	prm["newton_solver"]["relative_tolerance"] = 1E-7
	prm["newton_solver"]["maximum_iterations"] = 25
	prm["newton_solver"]["relaxation_parameter"] = 1.0
	prm["newton_solver"]["preconditioner"] = "default"
	prm["newton_solver"]["linear_solver"] = "mumps"


	#set_log_level(PROGRESS)

	solver_alpha = PETScTAOSolver()


	alpha_lb = interpolate(Expression("0.", degree =1), V_alpha) # lower bound, set to 0
	alpha_ub = interpolate(Expression("1.", degree =1), V_alpha) # upper bound, set to 1

	for bc in bc_alpha:
		bc.apply(alpha_lb.vector())

	for bc in bc_alpha:
		bc.apply(alpha_ub.vector())
	#=======================================================================================
	# To store results
	#=======================================================================================
	results = []
	file_alpha = File(save_dir+"/alpha.pvd") # use .pvd if .xdmf is not working
	file_u = File(save_dir+"/u.pvd") # use .pvd if .xdmf is not working

	#=======================================================================================
	# Solving at each timestep
	#=======================================================================================
	load_multipliers = np.linspace(pressure_min,pressure_max,pressure_steps)
	energies = np.zeros((len(load_multipliers),5))
	iterations = np.zeros((len(load_multipliers),2))
	forces = np.zeros((len(load_multipliers),2))

	for (i_t, t) in enumerate(load_multipliers):

	    print"\033[1;32m--- Time step %d: t = %g ---\033[1;m" % (i_t, t)
	    #u_T.t = t*ut
	    #u_B.t = t*ut
	    p_b.t = t
	    Pressure.assign(interpolate(p_b, V_alpha))
	    # Alternate mininimization
	    # Initialization
	    iter = 1; err_alpha = 1
	    # Iterations
	    while err_alpha>tolerance and iter<max_iterations:
		# solve elastic problem
		solver_u.solve()
		# solve damage problem
     		solver_alpha.solve(problem_alpha, alpha_.vector(), alpha_lb.vector(), alpha_ub.vector())

	       	# test error
	       	err_alpha = (alpha_.vector() - alpha_0.vector()).norm('linf')
		# monitor the results
	       	if mpi_comm_world().rank == 0:
		    print "Iteration:  %2d, Error: %2.8g, alpha_max: %.8g" %(iter, err_alpha, alpha_.vector().max())
		
		# update iteration
		alpha_0.vector()[:] = alpha_.vector()
		iter += 1

	    # Update the lower bound to account for the irreversibility
	    alpha_lb.vector()[:] = alpha_.vector()
	    
	    ## plot the damage fied
	    #plt.figure(2)	
	    #plot(alpha_, range_min = 0., range_max = 1., key = "alpha", title = "Damage at loading %.4f"%(ut*t)) 
	    #plt.show()   
	    #plt.interactive(True)
	    #=======================================================================================
	    # Some post-processing
	    #=======================================================================================
	    # Save number of iterations for the time step    
	    iterations[i_t] = np.array([t,iter])
	    # Calculate the energies
	    elastic_energy_value = assemble(elastic_energy)
	    dissipated_energy_value = assemble(dissipated_energy)
	    pressurized_energy_value = assemble(pressurized_energy)
	    if mpi_comm_world().rank == 0:
		print("\nEnd of timestep %d with load multiplier %g"%(i_t, t))
		print("AM: Iteration number: %i - Elastic_energy: %.3e" % (i_t, elastic_energy_value))
		print("AM: Iteration number: %i - Dissipated_energy: %.3e" % (i_t, dissipated_energy_value))
		print("AM: Iteration number: %i - pressurized_energy: %.3e" % (i_t, pressurized_energy_value))
		print"\033[1;32m--------------------------------------------------------------\033[1;m"

	    energies[i_t] = np.array([t,elastic_energy_value,dissipated_energy_value,pressurized_energy_value,elastic_energy_value+dissipated_energy_value+pressurized_energy_value])
	    # Calculate the axial force resultant 
	    forces[i_t] = np.array([t,assemble(inner(sigma(u_,alpha_)*e1,e1)*ds(1))])
	    # Dump solution to file 
	    file_alpha << (alpha_,t) 
	    file_u << (u_,t) 
	    # Save some global quantities as a function of the time 
	    np.savetxt(save_dir+'/energies.txt', energies)
	    np.savetxt(save_dir+'/forces.txt', forces)
	    np.savetxt(save_dir+'/iterations.txt', iterations)


	#=======================================================================================
	# Save alpha and displacement data
	#=======================================================================================
	output_file_u = HDF5File(mpi_comm_world(), save_dir+"u_4_opening.h5", "w") # self.save_dir + "uO.h5"
	output_file_u.write(u_, "solution")
	output_file_u.close()

	output_file_alpha = HDF5File(mpi_comm_world() , save_dir+"alpha_4_opening.h5", "w")
	output_file_alpha.write(alpha_, "solution")
	output_file_alpha.close()


if __name__ == '__main__':
        Fracking()


