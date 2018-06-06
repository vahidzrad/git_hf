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



#=======================================================================================
# Input date
#=======================================================================================
# Geometry
L = 4.0 # length
H = 4.0 # height
hsize= 0.02 # target cell size
meshname="diffusion_hsize%g" % (hsize)

# Stopping criteria for the alternate minimization
max_iterations = 200
tolerance = 1.0e-5

# Loading
body_force = Constant((0.,0.))  # bulk load
pressure_min = 0. # load multiplier min value
pressure_max = 10. # load multiplier max value
pressure_steps = 10 # number of time steps


#====================================================================================
#Input data  pressure field
#====================================================================================
biot=0.75
phi=0.01 #porosity


kappa= 9.8e-9 #is the permeability of the rock
mu_dynamic= 0.0098  #is the dynamic viscosity of the fluid

f = Constant(0)

Init_Pressure=0.1

Q=1.e-3
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


mesh = Mesh('meshes/diffusion_hsize'+str(float(hsize))+'.xml')
mesh_fun = MeshFunction("size_t", mesh,"meshes/diffusion_hsize"+str(float(hsize))+"_facet_region.xml")
plt.figure(1)
plot(mesh, "2D mesh")
plt.interactive(True)

ndim = mesh.geometry().dim() # get number of space dimensions

#=======================================================================================
# others definitions
#=======================================================================================
prefix = "L%s-H%.2f"%(L,H)
save_dir = "GasDiffusion_result/" + prefix + "/"

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
V_p = FunctionSpace(mesh, "CG", 1)

# Define the function, test and trial fields
P_, P, P_t= Function(V_p), TrialFunction(V_p), TestFunction(V_p),


rho = Function(V_p)  #density kg/m^3
N = Function(V_p)    #N=R*T*(delta**2 * phi_r_delta_delta +2*delta*phi_r_delta + 1) # J/(Kg*K)*K-->m^2/s^2

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
## bc - P (imposed pressure)
P_C = Expression("p", p=0.0, degree=1)
Q = Expression("p*DeltaT/t_stop*Q", p=0.0, Q=Q, DeltaT=DeltaT, t_stop=t_stop, degree=1)

Gamma_P_0 = DirichletBC(V_p, 0.0, boundaries, 1)
Gamma_P_1 = DirichletBC(V_p, 0.0, boundaries, 2)
Gamma_P_2 = DirichletBC(V_p, 0.0, boundaries, 3)
Gamma_P_3 = DirichletBC(V_p, 0.0, boundaries, 4)
#Gamma_P_4 = DirichletBC(V_p, P_C, mesh_fun, 101)
#Gamma_P_4 = DirichletBC(V_p, P_C, boundaries, 5)
bc_P = [Gamma_P_0, Gamma_P_1, Gamma_P_2, Gamma_P_3]#, Gamma_P_4]
#====================================================================================
# Define  problem and solvers
#====================================================================================
Pressure = phi/N*P_*P*dx + DeltaT*(kappa/mu_dynamic)*rho*inner(nabla_grad(P_), nabla_grad(P))*dx-(phi/N*P_0 +DeltaT*f)*P*dx-rho*DeltaT*Q*P*ds(5) 
#+kappa* dot(P, Flux_T)*ds(3) +kappa* dot(P, Flux_B) *ds(4) + kappa*dot(Flux_C, P)* ds(5) 

#------------------------------------------------------------------------------------


#Jacobian of pressure problem
J_p  = derivative(Pressure, P_, P_t) 

problem_pressure = NonlinearVariationalProblem(Pressure, P_, bc_P, J=J_p)
solver_pressure = NonlinearVariationalSolver(problem_pressure) 
#=======================================================================================
# To store results
#=======================================================================================
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
    #P_C.p = p
    Q.p = p

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


    
    plt.figure(2)	
    plot(P_, key = "P", title = "Pressure %.4f"%(p*pressure_max)) 
    plt.show()   
    plt.interactive(True)
    
    
    file_p << (P_,p) 

