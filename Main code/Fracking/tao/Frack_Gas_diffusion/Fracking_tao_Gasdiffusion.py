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
from EOS import EqOfState
from EOS_N import EqOfState_N
from decimal import *

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
# Input date
#=======================================================================================
# Geometry
L = 4.0 # length
H = 4.0 # height
hsize= 0.02 # target cell size
meshname="fracking_hsize%g" % (hsize)

# Material constants
ell = Constant(2 * hsize) # internal length scale
E = 6.e3 # Young modulus
nu = 0.3 # Poisson ratio

biot= 0. #Biot coefficient

PlaneStress= False

gc = 1. # fracture toughness
k_ell = Constant(1.0e-12) # residual stiffness
law = "AT1"

# effective toughness 
if law == "AT2":  
	Gc = gc/(1+hsize/(2*ell)) #AT2
elif  law == "AT1": 
	Gc = gc/(1+3*hsize/(8*ell)) #AT1
else:
	Gc = gc 


ModelB= False 

# Stopping criteria for the alternate minimization
max_iterations = 200
tolerance = 1.0e-5

# Loading
ut = 1. # reference value for the loading (imposed displacement)
body_force = Constant((0.,0.))  # bulk load
pressure_min = 0. # load multiplier min value
pressure_max = 300. # load multiplier max value
pressure_steps = 10 # number of time steps

WheelerApproach= True
#====================================================================================
#Input data  pressure field
#====================================================================================
biot=0.75
phi=0.01 #porosity


kappa= 9.8e-9 #is the permeability of the rock
mu_dynamic= 0.0098  #is the dynamic viscosity of the fluid

f = Constant(0)

Init_Pressure=0.5
#=======================================================================================
# Geometry and mesh generation
#=======================================================================================
# Generate a XDMF/HDF5 based mesh from a Gmsh string
geofile = \
		"""
		lc = DefineNumber[ %g, Name "Parameters/lc" ];
		H = 4;
		L = 4;

		r=0.1;
		a=0.05;

                Point(1) = {0, 0, 0, 1*lc};
                Point(2) = {L, 0, 0, 1*lc};
                Point(3) = {L, H, 0, 1*lc};
                Point(4) = {0, H, 0, 1*lc};

                Point(5) = {L/2, H/2+r+a, 0, 1*lc};
                Point(6) = {L/2, H/2+r, 0, 1*lc};

                Point(7) = {L/2, H/2-r, 0, 1*lc};
                Point(8) = {L/2, H/2-r-a, 0, 1*lc};

                Point(9) = {L/2, H/2, 0, 1*lc};

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
		Physical Line(103) = {8};
		Physical Line(104) = {9};

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
	return  0.5*(lmbda+2/3*mu) * ( angle_bracket_plus(tr(dev_eps(u_))**2)) + mu*dev_eps(u_)**2 + 0.5*(lmbda+2/3*mu) * ( angle_bracket_minus(tr(dev_eps(u_))**2))
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
prefix = "%s-L%s-H%.2f-S%.4f-l%.4f"%(law,L,H,hsize, ell)
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

class Void(SubDomain):
    def inside(self, x, on_boundary):
	return   x[0] <= 2.5 and x[0] >= 1.5  and  x[1] <= 2.5 and x[1] >= 1.5 and on_boundary

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
V_alpha = FunctionSpace(mesh, "CG", 1)
V_p = FunctionSpace(mesh, "CG", 1)

# Define the function, test and trial fields
u_, u, u_t = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
alpha_, alpha, alpha_t = Function(V_alpha), TrialFunction(V_alpha), TestFunction(V_alpha)
P_, P, P_t= Function(V_p), TrialFunction(V_p), TestFunction(V_p),


rho = Function(V_p)  #density kg/m^3
N = Function(V_p)    #N=R*T*(delta**2 * phi_r_delta_delta +2*delta*phi_r_delta + 1) # J/(Kg*K)*K-->m^2/s^2

alpha_0 = Function(V_alpha)
#define_alpha_0=Expression("x[1] == 2. & x[0] <= 2.2 & x[0] >=1.8 ? 1.0 : 0.0", degree=1)
#alpha_0.interpolate(define_alpha_0) #added by Mostafa
alpha_0 = interpolate(Constant(0.0), V_alpha) # initial (known) alpha, undamaged everywhere.
P_0 = interpolate(Expression('Ini_P', Ini_P=Init_Pressure, degree=1), V_p)
#=======================================================================================
# Define initial time, density and N
#=======================================================================================
Rho_intial=EqOfState(Init_Pressure) #initial density according to initial pressure(Pa?)
N_intial=EqOfState_N([Rho_intial]) #initial N(m^2/s^2) according to initial density(kg/m^3)
#-----------------------
t_stop = H**2 *mu_dynamic/(kappa*Rho_intial*N_intial[0])  # non dimensional time (s) 
DeltaT=t_stop/pressure_steps 

print "Rho_intial=",Rho_intial
print "N_intial=",N_intial[0]
print  "t_stop=",t_stop
#------------------------------------------------------------------------------------------
N_intial=Constant(N_intial[0])
t_stop_a=Constant(t_stop)
Rho_intial=Constant(Rho_intial)

rho = interpolate(Expression('Rho', Rho = Rho_intial, degree=1), V_p)
N = interpolate(Expression('N', N= N_intial, degree=1), V_p)
#=======================================================================================
# Dirichlet boundary condition for a traction test boundary
#=======================================================================================
# bc - u (imposed displacement)
u_R = zero_v #Expression(("p", "0.",), p=0.0, degree=1)
u_L = zero_v
u_T = zero_v#Expression(("0.", "p",), p=0.0, degree=1)
u_B = zero_v#Expression(("0.", "-p",), p=0.0, degree=1)

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
Gamma_alpha_5 = DirichletBC(V_alpha, 1.0, mesh_fun, 102)
bc_alpha = [Gamma_alpha_0, Gamma_alpha_1, Gamma_alpha_2, Gamma_alpha_3,Gamma_alpha_4, Gamma_alpha_5]

## bc - P (imposed pressure)
P_C = Expression("p", p=0.0, degree=1)

Gamma_P_0 = DirichletBC(V_p, 0.0, boundaries, 1)
Gamma_P_1 = DirichletBC(V_p, 0.0, boundaries, 2)
Gamma_P_2 = DirichletBC(V_p, 0.0, boundaries, 3)
Gamma_P_3 = DirichletBC(V_p, 0.0, boundaries, 4)
#Gamma_P_4 = DirichletBC(V_p, P_C, mesh_fun, 101)
Gamma_P_4 = DirichletBC(V_p, P_C, boundaries, 5)
bc_P = [Gamma_P_0, Gamma_P_1, Gamma_P_2, Gamma_P_3, Gamma_P_4]
#====================================================================================
# Define  problem and solvers
#====================================================================================
Pressure = phi/N*P_*P*dx + DeltaT*(kappa/mu_dynamic)*rho*inner(nabla_grad(P_), nabla_grad(P))*dx-(phi/N*P_0 +DeltaT*f)*P*dx
	#+kappa* dot(P, Flux_T)*ds(3) +kappa* dot(P, Flux_B) *ds(4) + kappa*dot(Flux_C, P)* ds(5) 

#------------------------------------------------------------------------------------
elastic_energy = psi(u_, alpha_)*dx
external_work = dot(body_force, u_)*dx

if not WheelerApproach:  # Bourdin approach, 2012 , SPE 159154, equation (5)
	pressurized_energy = P_ * inner(u_, grad(alpha_)) *dx
else:  # M.F. Wheeler et al., CAMAME, 2014, eqution (4)
	pressurized_energy = -(biot-1.)*((1.-alpha_)**2 * P_ * div(u_)) * dx+ dot((1-alpha_)**2 * grad(P_), u_) * dx

dissipated_energy = Gc/float(c_w)*( w(alpha_)/ell + ell* inner( grad(alpha_), grad(alpha_)) )*dx 

total_energy = elastic_energy + dissipated_energy + pressurized_energy - external_work



# Residual and Jacobian of elasticity problem
Du_total_energ = derivative(total_energy, u_, u_t)
J_u = derivative(Du_total_energ, u_, u)

# Residual and Jacobian of damage problem
Dalpha_total_energy = derivative(total_energy, alpha_, alpha_t)
J_alpha = derivative(Dalpha_total_energy, alpha_, alpha)


#Jacobian of pressure problem
J_p  = derivative(Pressure, P_, P_t) 

# Variational problem for the displacement
problem_u = NonlinearVariationalProblem(Du_total_energ, u_, bc_u, J_u)


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

solver_alpha = PETScTAOSolver()


alpha_lb = interpolate(Expression("0.", degree =1), V_alpha) # lower bound, set to 0
alpha_ub = interpolate(Expression("1.", degree =1), V_alpha) # upper bound, set to 1

for bc in bc_alpha:
	bc.apply(alpha_lb.vector())

for bc in bc_alpha:
	bc.apply(alpha_ub.vector())


problem_pressure = NonlinearVariationalProblem(Pressure, P_, bc_P, J=J_p)
solver_pressure = NonlinearVariationalSolver(problem_pressure) 
#=======================================================================================
# To store results
#=======================================================================================
results = []
file_alpha = File(save_dir+"/alpha.pvd") # use .pvd if .xdmf is not working
file_u = File(save_dir+"/u.pvd") # use .pvd if .xdmf is not working
file_p = File(save_dir+"/p.pvd") 
#=======================================================================================
# Solving at each timestep
#=======================================================================================
load_multipliers = np.linspace(pressure_min,pressure_max,pressure_steps)
energies = np.zeros((len(load_multipliers),5))
iterations = np.zeros((len(load_multipliers),2))
forces = np.zeros((len(load_multipliers),2))

for (i_p, p) in enumerate(load_multipliers):

    print"\033[1;32m--- Time step %d: time = %g, pressure_max = %g ---\033[1;m" % (i_p, i_p*DeltaT, p)
    #u_R.p = p*ut
    #u_B.t = t*ut
    P_C.p = p
	

    iteration = 0; iter= 0 ; err_P = 1;  err_alpha = 1
    # Iterations

    while  err_P>tolerance and iteration<max_iterations:
    	# solve pressure problem
	solver_pressure.solve()
	print P_
	

        rho=EqOfState(P_.vector().get_local().shape) 	#set the new density according new pressure
        N=EqOfState_N(rho)			#set the new N according new pressure

     	err_P = (P_.vector() - P_0.vector()).norm('linf')
 	if mpi_comm_world().rank == 0:
        	print "Iteration:  %2d, pressure_Error: %2.8g, P_max: %.8g" %(iteration, err_P, P_.vector().max())
        P_0.vector()[:] = P_.vector()
        iteration += 1


    while err_alpha>tolerance and iter<max_iterations:
     	# solve elastic problem
        solver_u.solve()
       	if mpi_comm_world().rank == 0:
        	print "Elastic iteration: %2d" % (iter)
        # solve damage problem
	solver_alpha.solve(problem_alpha, alpha_.vector(), alpha_lb.vector(), alpha_ub.vector())
       	# test error
       	err_alpha = (alpha_.vector() - alpha_0.vector()).norm('linf')
        # monitor the results
       	if mpi_comm_world().rank == 0:
        	print "Alpha iteration: %2d, Error: %2.8g" % (iter, err_alpha)

        # update iteration
        alpha_0.vector()[:] = alpha_.vector()
        iter += 1
    
    # Update the lower bound to account for the irreversibility
    alpha_lb.vector()[:] = alpha_.vector()


    plt.figure(2)	
    plot(P_, key = "P", title = "Pressure %.4f"%(p*pressure_max)) 
    plt.show()   
    plt.interactive(True)
    
    # plot the damage fied
    plt.figure(3)	
    plot(alpha_, range_min = 0., range_max = 1., key = "alpha", title = "Damage at loading %.4f"%(p*pressure_max)) 
    plt.show()   
    plt.interactive(True)
    #=======================================================================================
    # Some post-processing
    #=======================================================================================
    # Save number of iterations for the time step    
    iterations[i_p] = np.array([p,iter])
    # Calculate the energies
    elastic_energy_value = assemble(elastic_energy)
    dissipated_energy_value = assemble(dissipated_energy)
    pressurized_energy_value = assemble(pressurized_energy)
    if mpi_comm_world().rank == 0:
        print("\nEnd of timestep %d with load multiplier %g"%(i_p, p))
        print("AM: Iteration number: %i - Elastic_energy: %.3e" % (i_p, elastic_energy_value))
        print("AM: Iteration number: %i - Dissipated_energy: %.3e" % (i_p, dissipated_energy_value))
        print("AM: Iteration number: %i - pressurized_energy: %.3e" % (i_p, pressurized_energy_value))
	print"\033[1;32m--------------------------------------------------------------\033[1;m"

    energies[i_p] = np.array([p,elastic_energy_value,dissipated_energy_value,pressurized_energy_value,elastic_energy_value+dissipated_energy_value+pressurized_energy_value])
    # Calculate the axial force resultant 
    forces[i_p] = np.array([p,assemble(inner(sigma(u_,alpha_)*e1,e1)*ds(1))])
    # Dump solution to file 
    file_alpha << (alpha_,p) 
    file_u << (u_,p) 
    file_p << (P_,p) 
    # Save some global quantities as a function of the time 
    np.savetxt(save_dir+'/energies.txt', energies)
    np.savetxt(save_dir+'/forces.txt', forces)
    np.savetxt(save_dir+'/iterations.txt', iterations)

#=======================================================================================
# Plot energy and stresses
#=======================================================================================
def critical_stress():
    xs = sympy.Symbol('x')
    wx = w(xs); sx = 1/(E*H*a(xs));
    res = sympy.sqrt(2*(Gc*H/c_w)*wx.diff(xs)/(sx.diff(xs)*ell))
    return res.evalf(subs={xs:0})

def plot_stress():
    plt.plot(forces[:,0], forces[:,1], 'b-o', linewidth = 2)
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    force_cr = critical_stress()
    plt.axvline(x = force_cr/(E*H)*L, color = 'grey', linestyle = '--', linewidth = 2)
    plt.axhline(y = force_cr, color = 'grey', linestyle = '--', linewidth = 2)

def plot_energy():
    p1, = plt.plot(energies[:,0], energies[:,1],'b-o',linewidth=2)
    p2, = plt.plot(energies[:,0], energies[:,2],'r-o',linewidth=2)
    p3, = plt.plot(energies[:,0], energies[:,3],'k--',linewidth=2)
    plt.legend([p1, p2, p3], ["Elastic","Dissipated","Total"])
    plt.xlabel('Displacement')
    plt.ylabel('Energies')
    force_cr = critical_stress()
    plt.axvline(x = force_cr/(E*H)*L, color = 'grey',linestyle = '--', linewidth = 2)
    plt.axhline(y = H,color = 'grey', linestyle = '--', linewidth = 2)

def plot_energy_stress():
    plt.subplot(211)
    plot_stress()
    plt.subplot(212)
    plot_energy()
    plt.savefig(save_dir+'/energies_force.png')
    plt.show()

plot_energy_stress()
plt.interactive(True)


#### Remove the .pyc file ####
MPI.barrier(mpi_comm_world())
if MPI.rank(mpi_comm_world()) == 0:
    os.remove("EOS.pyc")
    os.remove("EOS_N.pyc")

