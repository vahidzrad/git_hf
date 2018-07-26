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
#from CrkOpn_MosVah import Opening
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
if not has_petsc():
    print("DOLFIN must be compiled at least with PETSc 3.6 to run this demo.")
    exit(0)


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
# Input date
#=======================================================================================
# Geometry
L = 1. # length
H = 1.0 # height
hsize= 1.0e-2 # target cell size
meshname="fracture_hsize%g" % (hsize)

# Material constants
ell = Constant(4* hsize) # internal length scale
#E = 6e3 # Young modulus MPa
#nu = 0.3 # Poisson ratio

PlaneStress= False

gc = 2.7e-3 #1.0 # fracture toughness MPa.m^0.5
k_ell = Constant(1.0e-12) # residual stiffness
law = "AT2"

# effective toughness 
if law == "AT2":  
	Gc = gc/(1+hsize/(2*ell)) #AT2
elif  law == "AT1": 
	Gc = gc/(1+3*hsize/(8*ell)) #AT1
else:
	Gc = gc #??


ModelB= False 
if not ModelB:  # Model A (isotropic model)
	Model = 'Isotropic'
else:  # Model B (Amor's model)
	Model = 'Amor'



# Stopping criteria for the alternate minimization
max_iterations = 100
tolerance = 1.0e-6

# Loading
ut = 6.e-3 # reference value for the loading (imposed displacement)
body_force = Constant((0.,0.))  # bulk load
load_min = 0. # load multiplier min value
load_max = 1. # load multiplier max value
load_steps = 10 # number of time steps

#=======================================================================================
# Geometry and mesh generation
#=======================================================================================
# Generate a XDMF/HDF5 based mesh from a Gmsh string
geofile = \
		"""
		lc = DefineNumber[ %g, Name "Parameters/lc" ];
		// Characteristic length made large so the number of elements is small

		H = 1;
		L = 1;

		// Point: X, Y, Z coordinates, & characteristic length setting elt size
		Point(1) = {0, 0, 0, lc} ;
		Point(2) = {L, 0, 0, lc} ;
		Point(3) = {L, H/2, 0, lc} ;
		Point(4) = {L, H, 0, lc} ;
		Point(5) = {0, H, 0, lc} ;
		Point(6) = {0, H/2, 0, lc} ;
		Point(7) = {L/2, H/2, 0, lc} ;


		// Lines defined by lists of numbers:
		Line(1) = {1, 2} ;
		Line(2) = {2, 3} ;
		Line(3) = {3, 7} ;
		Line(4) = {7, 6} ;
		Line(5) = {6, 1} ;

		Line(6) = {6, 5} ;
		Line(7) = {5, 4} ;
		Line(8) = {4, 3} ;

		// Line loops to construct surfaces
		Line Loop(1) = {1, 2, 3, 4, 5} ;
		Line Loop(2) = {6, 7, 8, 3, 4} ;

		// Surfaces for meshing
		Plane Surface(1) = {1} ;
		Plane Surface(2) = {2} ;

		Physical Point(13) = {6, 7};
		Physical Line(105) = {4} ;
		Physical Surface(201) = {1} ;
		Physical Surface(202) = {2} ;

		// End of file: crack2.geo

		// File: crack2a.geo, for post-processing crack2.msh

		// Duplicate nodes around Line 5; a mock crack
		Plugin(Crack).Dimension = 1 ;
		Plugin(Crack).PhysicalGroup = 105 ;
		Plugin(Crack).OpenBoundaryPhysicalGroup = 13 ;
		Plugin(Crack).Run ;

		// End of file: crack2a.geo


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


mesh = Mesh('meshes/fracture_hsize'+str(float(hsize))+'.xml')
mesh_fun = MeshFunction("size_t", mesh,"meshes/fracture_hsize"+str(float(hsize))+"_facet_region.xml")
plt.figure(1)
plot(mesh, "2D mesh")
plt.interactive(True)

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
	K = lmbda+2/3*mu
	return  g(alpha_) * ( K * ( angle_bracket_plus( tr(eps(u_))) * Identity(ndim) )+ 2*mu*dev_eps(u_) ) \
		+ K*( angle_bracket_minus(tr(dev_eps(u_))) * Identity(ndim))
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
	K = lmbda+2/3*mu
	return  g(alpha_) * ( 0.5*K * ( angle_bracket_plus(tr(dev_eps(u_))))**2 + mu*dev_eps(u_)**2) + 0.5*K * ( angle_bracket_minus(tr(dev_eps(u_))))**2
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
save_dir = "Fracture_QS_result/" + prefix + "/"

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
	lmbda = 121.15# E*nu/((1.0+nu)*(1.0-2.0*nu))
else:  # plane stress
	lmbda = 121.15# E*nu/(1.0-nu**2)

# shear modulus
mu =80.77# E / (2.0 * (1.0 + nu)) 
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

alpha_0 = Function(V_alpha)
#define_alpha_0=Expression("x[1] == 0.5 & x[0] <= 0.5 & x[0] >=0 ? 1.0 : 0.0", degree=1)
#alpha_0.interpolate(define_alpha_0) #added by Mostafa
alpha_0 = interpolate(Constant(0.0), V_alpha) # initial (known) alpha, undamaged everywhere.
#=======================================================================================
# Dirichlet boundary condition for a traction test boundary
#=======================================================================================
# bc - u (imposed displacement)
u_R = zero_v#Expression(("t", "0.",), t=0.0, degree=1)
u_L = zero_v
u_T = Expression(("0.", "t",), t=0.0, degree=1)
u_B = zero_v#Expression(("0.", "-t",), t=0.0, degree=1)

Gamma_u_0 = DirichletBC(V_u, u_R, boundaries, 1)
Gamma_u_1 = DirichletBC(V_u, u_L, boundaries, 2)
Gamma_u_2 = DirichletBC(V_u, u_T, boundaries, 3)
Gamma_u_3 = DirichletBC(V_u, u_B, boundaries, 4)
bc_u = [ Gamma_u_2, Gamma_u_3]

# bc - alpha (zero damage)
Gamma_alpha_0 = DirichletBC(V_alpha, Constant(0.0), boundaries, 1)
Gamma_alpha_1 = DirichletBC(V_alpha, Constant(0.0), boundaries, 2)
Gamma_alpha_2 = DirichletBC(V_alpha, Constant(0.0), boundaries, 3)
Gamma_alpha_3 = DirichletBC(V_alpha, Constant(0.0), boundaries, 4)
#Gamma_alpha_4 = DirichletBC(V_alpha, Constant(1.0), mesh_fun, 101)
bc_alpha = [Gamma_alpha_0, Gamma_alpha_1, Gamma_alpha_2, Gamma_alpha_3]#, Gamma_alpha_4]

# bc - sigma (imposed traction)
sigma_T = Expression(("0.", "t",), t=0.0, degree=1) #traction on TOP boundary
sigma_B = Expression(("0.", "-t",), t=0.0, degree=1) #traction on TOP boundary

#====================================================================================
# Define  problem and solvers
#====================================================================================
elastic_energy = psi(u_, alpha_)*dx
external_work = dot(body_force, u_)*dx #+ dot(sigma_T, u_)*ds(3)+ dot(sigma_B, u_)*ds(4)
dissipated_energy = Gc/float(c_w)*(w(alpha_)/ell + ell*inner(grad(alpha_), grad(alpha_)))*dx 

print c_w
total_energy = elastic_energy + dissipated_energy - external_work

# First derivatives of energies (Residual)
Du_total_energy = derivative(total_energy, u_, u_t)
Dalpha_total_energy = derivative(total_energy, alpha_, alpha_t)

# Second derivatives of energies (Jacobian)
J_u = derivative(Du_total_energy, u_, u)
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


#-------------------
# Solver setup
#-------------------
# Set up the solvers  
solver_u = NonlinearVariationalSolver(problem_u)                 
prm = solver_u.parameters
prm["newton_solver"]["absolute_tolerance"] = 1E-6
prm["newton_solver"]["relative_tolerance"] = 1E-6
prm["newton_solver"]["maximum_iterations"] = 100
prm["newton_solver"]["relaxation_parameter"] = 1.0
prm["newton_solver"]["preconditioner"] = "default"
prm["newton_solver"]["linear_solver"] = "mumps"
#set_log_level(PROGRESS)


solver_alpha = PETScTAOSolver()


alpha_lb = interpolate(Expression("0.", degree =1), V_alpha) # lower bound, set to 0
alpha_ub = interpolate(Expression("1.", degree =1), V_alpha) # upper bound, set to 1
#info(solver_alpha.parameters,True) # uncomment to see available parameters
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
load_multipliers = np.linspace(load_min,load_max,load_steps)
energies = np.zeros((len(load_multipliers),4))
iterations = np.zeros((len(load_multipliers),2))
forces = np.zeros((len(load_multipliers),2))

for (i_t, t) in enumerate(load_multipliers):

    print"\033[1;32m--- Time step %d: t = %g ---\033[1;m" % (i_t, t)
    u_T.t = t*ut
    u_B.t = t*ut
    #sigma_T.t = t
    #sigma_B.t = t

    # Alternate mininimization
    # Initialization
    iter = 1; err_alpha = 1
    # Iterations
    while err_alpha>tolerance and iter<max_iterations:
        # solve elastic problem
        solver_u.solve()

        plt.figure(2)	
        plot(u_, title = "Displacement %.4f"%(ut*t)) 
        plt.show() 

        # solve damage problem
        solver_alpha.solve(problem_alpha, alpha_.vector(), alpha_lb.vector(), alpha_ub.vector())

        # plot the damage fied
        plt.figure(3)	
        plot(alpha_, range_min = 0., range_max = 1., key = "alpha", title = "Damage at loading %.4f"%(ut*t)) 
        plt.show()  

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
    
 
    plt.interactive(True)
    #=======================================================================================
    # Some post-processing
    #=======================================================================================
    # Save number of iterations for the time step    
    iterations[i_t] = np.array([t,iter])
    # Calculate the energies
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    if mpi_comm_world().rank == 0:
        print("\nEnd of timestep %d with load multiplier %g"%(i_t, t))
        print("AM: Iteration number: %i - Elastic_energy: %.3e" % (i_t, elastic_energy_value))
        print("AM: Iteration number: %i - Dissipated_energy: %.3e" % (i_t, surface_energy_value))
	print"\033[1;32m--------------------------------------------------------------\033[1;m"

    energies[i_t] = np.array([t,elastic_energy_value,surface_energy_value,elastic_energy_value+surface_energy_value])
    # Calculate the axial force resultant 
    forces[i_t] = np.array([ut*t,assemble(inner(sigma(u_,alpha_)*e1,e1)*ds(3))])
    # Dump solution to file 
    file_alpha << (alpha_,t) 
    file_u << (u_,t) 
    # Save some global quantities as a function of the time 
    np.savetxt(save_dir+'/energies.txt', energies)
    np.savetxt(save_dir+'/forces.txt', forces)
    np.savetxt(save_dir+'/iterations.txt', iterations)


#=======================================================================================
# Plot energy and stresses
#=======================================================================================
#def critical_stress():
#    xs = sympy.Symbol('x')
#    wx = w(xs); sx = 1/(E*H*a(xs));
#    res = sympy.sqrt(2*(Gc*H/c_w)*wx.diff(xs)/(sx.diff(xs)*ell))
#    return res.evalf(subs={xs:0})

def plot_stress():
    plt.plot(forces[:,0], forces[:,1], 'b-o', linewidth = 2)
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    #force_cr = critical_stress()
    #plt.axvline(x = force_cr/(E*H)*L, color = 'grey', linestyle = '--', linewidth = 2)
    #plt.axhline(y = force_cr, color = 'grey', linestyle = '--', linewidth = 2)

def plot_energy():
    p1, = plt.plot(energies[:,0], energies[:,1],'b-o',linewidth=2)
    p2, = plt.plot(energies[:,0], energies[:,2],'r-o',linewidth=2)
    p3, = plt.plot(energies[:,0], energies[:,3],'k--',linewidth=2)
    plt.legend([p1, p2, p3], ["Elastic","Dissipated","Total"])
    plt.xlabel('Displacement')
    plt.ylabel('Energies')
    #force_cr = critical_stress()
    #plt.axvline(x = force_cr/(E*H)*L, color = 'grey',linestyle = '--', linewidth = 2)
    #plt.axhline(y = H,color = 'grey', linestyle = '--', linewidth = 2)

def plot_energy_stress():
    plt.subplot(211)
    plot_stress()
    plt.subplot(212)
    plot_energy()
    plt.savefig(save_dir+'/energies_force.png')
    plt.show()

plot_energy_stress()
plt.interactive(True)
#=======================================================================================
# Save alpha and displacement data
#=======================================================================================
output_file_u = HDF5File(mpi_comm_world(), save_dir+"u_4_opening.h5", "w") # self.save_dir + "uO.h5"
output_file_u.write(u_, "solution")
output_file_u.close()

output_file_alpha = HDF5File(mpi_comm_world() , save_dir+"alpha_4_opening.h5", "w")
output_file_alpha.write(alpha_, "solution")
output_file_alpha.close()

# Project and write stress field to post-processing file
s = sigma_A(u_, alpha_) - (1./3)*tr(sigma_A(u_, alpha_))*Identity(ndim)  # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))
V = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, V)
File(save_dir+"von_Mises.pvd") << von_Mises

#arr_Coor_plt_X, arr_li, Volume = Opening(hsize)
