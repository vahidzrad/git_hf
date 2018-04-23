# FEnics code  Variational Fracture Mechanics
#

# A static solution of the variational fracture mechanics problems using the regularization AT2
#
# author: corrado.maurini@upmc.fr
#
# date: 25/05/2013
#
from dolfin import *
from mshr import *
import sys, os, sympy, shutil, math
import numpy as np


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
#-----------------------------------------------------------------------------------------
# Parameters
#-----------------------------------------------------------------------------------------
set_log_level(ERROR) # log level
parameters.parse()   # read paramaters from command line


# parameters of the nonlinear solver used for the alpha-problem

snes_solver_parameters_bounds = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "lu",
                                          "maximum_iterations": 50,
                                          "report": True,
                                          "error_on_nonconvergence": False}}


solver_u_parameters =  {"linear_solver" : "tfqmr", 
                            "symmetric" : False, 
                            "preconditioner" : "hypre_amg", 
                            "krylov_solver" : {
                                "report" : False,
                                "monitor_convergence" : False,
                                "relative_tolerance" : 1e-8 
                                }
                            }



# Geometry
# Geometry
L = 4.0 # length
H = 2.0 # height
cell_size= 0.01 # target cell size
meshname="fracture_hsize%g" % (cell_size)

# Material constant
E, nu = Constant(10.0), Constant(0.3)
Gc = Constant(1.0)
ellv = 4*cell_size; ell = Constant(ellv)
k_ell = Constant(1.e-6) # residual stiffness
# Loading
ut = 0.5 # reference value for the loading (imposed displacement)
f = 0. # bulk load
load_min = 0. # load multiplier min value
load_max = 1. # load multiplier max value
load_steps = 10 # number of time steps
# Numerical parameters of the alternate minimization
maxiter = 200 
toll = 1e-5
# Constitutive functions of the damage model
def w(alpha):
    return alpha

def a(alpha):
    return (1-alpha)**2

modelname = "AT1"
# others


savedir = "results/%s-bar-L%s-H%.2f-S%.4f-l%.4f"%(modelname,L,H,cell_size,ellv)
if os.path.isdir(savedir):
    shutil.rmtree(savedir)
# ------------------
# Geometry and mesh generation
# ------------------
# Generate a XDMF/HDF5 based mesh from a Gmsh string
geofile = \
		"""
		lc = DefineNumber[ %g, Name "Parameters/lc" ];
		H = 2;
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

"""%(cell_size)


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


mesh = Mesh('meshes/fracture_hsize'+str(float(cell_size))+'.xml')
mesh_fun = MeshFunction("size_t", mesh,"meshes/fracture_hsize"+str(float(cell_size))+"_facet_region.xml")
plt.figure(1)
plot(mesh, "2D mesh")
plt.interactive(True)

ndim = mesh.geometry().dim() # get number of space dimensions

#-------------------
# Useful definitions
#-------------------
# zero and unit vectors
zero_v = Constant((0.,)*ndim)
e1 = [Constant([1.,0.]),Constant((1.,0.,0.))][ndim-2]

 # Strain and stress
def eps(v):
    return sym(grad(v))
    
def sigma_0(v):
    mu    = E/(2.0*(1.0 + nu))
    lmbda = E*nu/(1.0 - nu**2)
    return 2.0*mu*(eps(v)) + lmbda*tr(eps(v))*Identity(ndim)
    
# Normalization constant for the dissipated energy 
# to get Griffith surface energy for ell going to zero
z = sympy.Symbol("z")
c_w = 2*sqrt(2)*sympy.integrate(sympy.sqrt(w(z)),(z,0,1))

#----------------------------------------------------------------------------
# Define boundary sets for boundary conditions
#----------------------------------------------------------------------------
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0] * 0.01, 0)
        
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[0] - L) * 0.01, 0.)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1] * 0.01, 0)
        
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[1] - H) * 0.01, 0.)
        
# Initialize sub-domain instances
left = Left() 
right = Right()
bottom=Bottom()
top=Top()

# define meshfunction to identify boundaries by numbers
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
left.mark(boundaries, 1) # mark left as 1
right.mark(boundaries, 2) # mark right as 2
bottom.mark(boundaries, 3) # mark right as 2
top.mark(boundaries, 4) # mark right as 2

# Define new measure including boundary naming 
ds = Measure("ds")[boundaries] # left: ds(1), right: ds(2)

#----------------------------------------------------------------------------
# Variational formulation 
#----------------------------------------------------------------------------
# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "CG", 1)
V_alpha = FunctionSpace(mesh, "CG", 1)

# Define the function, test and trial fields
u, du, v = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
alpha, dalpha, beta = Function(V_alpha), TrialFunction(V_alpha), TestFunction(V_alpha)

alpha_0 = interpolate(Expression("0.0", degree=1), V_alpha) # initial (known) alpha

# Dirichlet boundary condition for a traction test boundary
u_B = Expression(("0.", "-t",), t=0.0, degree=1)
u_T = Expression(("0.", "t",), t=0.0, degree=1)

# bc - u (imposed displacement)
Gamma_u_0 = DirichletBC(V_u, u_B, boundaries, 3)
Gamma_u_1 = DirichletBC(V_u, u_T, boundaries, 4)
bc_u = [Gamma_u_0, Gamma_u_1]
# bc - alpha (zero damage)
Gamma_alpha_0 = DirichletBC(V_alpha, 0.0, boundaries, 1)
Gamma_alpha_1 = DirichletBC(V_alpha, 0.0, boundaries, 2)
Gamma_alpha_2 = DirichletBC(V_alpha, 0.0, boundaries, 3)
Gamma_alpha_3 = DirichletBC(V_alpha, 0.0, boundaries, 4)
Gamma_alpha_4 = DirichletBC(V_alpha, 1.0, mesh_fun, 101)
bc_alpha = [Gamma_alpha_0, Gamma_alpha_1,  Gamma_alpha_2 , Gamma_alpha_3, Gamma_alpha_4]

# Fenics forms for the energies
def sigma(u,alpha):
    return (a(alpha)+k_ell)*sigma_0(u)
    
body_force = Constant((0.,0.))

elastic_energy = 0.5*inner(sigma(u,alpha), eps(u))*dx
external_work = dot(body_force, u)*dx
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell+ 0.5*ell*dot(grad(alpha), grad(alpha)))*dx 
total_energy = elastic_energy + dissipated_energy - external_work

# Weak form of elasticity problem
E_u = derivative(total_energy,u,v)
E_alpha = derivative(total_energy,alpha,beta)
E_alpha_alpha = derivative(E_alpha,alpha,dalpha)

# Writing tangent problems in term of test and trial functions for matrix assembly
E_du = replace(E_u,{u:du})
E_dalpha = replace(E_alpha,{alpha:dalpha})

# Variational problem for the displacement
problem_u = LinearVariationalProblem(lhs(E_du), rhs(E_du), u, bc_u)


# Set up the solvers                                        
solver_u = LinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)

#solver_alpha.parameters.update(solver_alpha_parameters)
lb = interpolate(Expression("0.", degree=1), V_alpha) # lower bound, set to 0
ub = interpolate(Expression("1.", degree=1), V_alpha) # upper bound, set to 1
#info(solver_alpha.parameters,True) # uncomment to see available parameters
problem_alpha_nl = NonlinearVariationalProblem(E_alpha, alpha, bc_alpha, J=E_alpha_alpha)
problem_alpha_nl.set_bounds(lb, ub)
solver_alpha = NonlinearVariationalSolver(problem_alpha_nl)
solver_alpha.parameters.update(snes_solver_parameters_bounds)



#  loading and initialization of vectors to store time datas
load_multipliers = np.linspace(load_min,load_max,load_steps)
energies = np.zeros((len(load_multipliers),4))
iterations = np.zeros((len(load_multipliers),2))
forces = np.zeros((len(load_multipliers),2))
file_alpha = File(savedir+"/alpha.pvd") # use .pvd if .xdmf is not working
file_u = File(savedir+"/u.pvd") # use .pvd if .xdmf is not working
# Solving at each timestep
alpha_error = Function(V_alpha)
for (i_t, t) in enumerate(load_multipliers):
    u_T.t = t*ut
    u_B.t = t*ut
    # Alternate mininimization
    # Initialization
    iter = 1; err_alpha = 1
    # Iterations
    while err_alpha>toll and iter<maxiter:
        # solve elastic problem
        solver_u.solve()
        # solve damage problem
        solver_alpha.solve()

        # test error
        alpha_error.vector()[:] = alpha.vector() - alpha_0.vector()
        err_alpha = np.linalg.norm(alpha_error.vector().array(), ord = np.Inf)
        #err_alpha = alphadiff.vector().max()
        # monitor the results
        if mpi_comm_world().rank == 0:
            print "Iteration:  %2d, Error: %2.8g, alpha_max: %.8g" %(iter, err_alpha, alpha.vector().max())
        # update iteration
        alpha_0.assign(alpha)
        iter=iter+1
    # plot the damage fied
    plt.figure(2)
    plot(alpha, range_min = 0., range_max = 1., key = "alpha", title = "Damage at loading %.4f"%(t*ut))
    plt.show()
    plt.interactive(False)
    # updating the lower bound to account for the irreversibility
    lb.vector()[:] = alpha.vector()
    # ----------------------------------------
    # Some post-processing
    # ----------------------------------------
    # Save number of iterations for the time step    
    iterations[i_t] = np.array([t,iter])
    # Calculate the energies
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    if mpi_comm_world().rank == 0:
        print("\nEnd of timestep %d with load multiplier %g"%(i_t, t))
        print("\nElastic and surface energies: (%g,%g)"%(elastic_energy_value,surface_energy_value))
        print "-----------------------------------------"
    energies[i_t] = np.array([t,elastic_energy_value,surface_energy_value,elastic_energy_value+surface_energy_value])
    # Calculate the axial force resultant 
    forces[i_t] = np.array([t,assemble(inner(sigma(u,alpha)*e1,e1)*ds(1))])
    # Dump solution to file 
    file_alpha << (alpha,t) 
    file_u << (u,t) 
    # Save some global quantities as a function of the time 
    np.savetxt(savedir+'/energies.txt', energies)
    np.savetxt(savedir+'/forces.txt', forces)
    np.savetxt(savedir+'/iterations.txt', iterations)

# Plot energy and stresses
import matplotlib.pyplot as plt
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
    plt.savefig(savedir+'/energies_force.png')
    plt.show()

plot_energy_stress()
interactive()
