
from dolfin import *
from mshr import *
import sys, os, sympy, shutil, math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
from numpy import loadtxt                                        
import matplotlib.pyplot as plt 
from EOS import EqOfState
from EOS_N import EqOfState_N
from decimal import *
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
getcontext().prec = 100
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
prefix = "L%s-H%s-S%s" % (L, H)
save_dir = "p_result/" + prefix + "/"

mesh = Mesh('Meshes/voidmesh.xml')

L = 4.0
H = 4.0

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
V_p = FunctionSpace(mesh, 'Lagrange', 1)
p_ = Function(V_p)   #pressure Pa
p_t = TestFunction(V_p)

rho = Function(V_p)  #density kg/m^3
N = Function(V_p)	 #N=R*T*(delta**2 * phi_r_delta_delta +2*delta*phi_r_delta + 1) # J/(Kg*K)*K-->m^2/s^2

#-----------------------------------------------------------------------------------------
# Define boundary sets for boundary conditions
#-----------------------------------------------------------------------------------------

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L, eps=100.0*DOLFIN_EPS) and on_boundary

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0, eps=100.0*DOLFIN_EPS) and on_boundary

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H, eps=100.0*DOLFIN_EPS) and on_boundary

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0, eps=100.0*DOLFIN_EPS) and on_boundary

# Todo: To make void more general, below use parameters rather than numbers
class Void(SubDomain):
    def inside(self, x, on_boundary):
        return   x[0] <= 2.5 and x[0] >= 1.5 and  x[1] <= 2.5 and x[1] >= 1.5 and on_boundary

# Initialize sub-domain instances
right = Right()
left = Left()
top = Top()
bottom = Bottom()

void=Void()

# define meshfunction to identify boundaries by numbers
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)

right.mark(boundaries, 1) # mark right as 1
left.mark(boundaries, 2) # mark left as 2
top.mark(boundaries, 3) # mark top as 3
bottom.mark(boundaries, 4) # mark bottom as 4

void.mark(boundaries, 5) # mark void as 5

# Define new measure including boundary naming 
ds = Measure("ds")[boundaries] # left: ds(1), right: ds(2)
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
pore = 0.01 #porosity
mu_dynamic = 4.4e-5 #dynamic viscosity, Pa.s
kappa=1.0e10 #permeability, m^2
Flux_T = Constant(0.)
Flux_B = Constant(0.)
Flux_C = Constant(0.)
f = Constant(0.)

rho_i = EqOfState(1e6) #initial density according to initial pressure(Pa)
print rho_i
N_i = EqOfState_N([0.0156654]) #initial N(m^2/s^2) according to initial density(kg/m^3)
print N_i
#------------------------------------------------------------------------------------------
n_iterations = 100 # number of time steps
# Question: Why 't_stop' is assigned twice?
t_stop = H**2 *mu_dynamic/(kappa*rho_i*N_i)  # non dimensional time (s)
t_stop = 7.03e-17
dt = t_stop/n_iterations
print  t_stop
#t_stop = H*mu_dynamic/kappa  
#------------------------------------------------------------------------------------------
rho = interpolate(Expression('Rho', Rho = 0.0156654, degree=1), V_p)
N = interpolate(Expression('N', N=63880.51497367, degree=1), V_p)
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
# Define boundary condition
P_R = Constant(1e6)
P_L = Constant(1e6)
P_B = Constant(1e6)
P_T = Constant(1e6)
P_V = Expression("t/t_stop*2e6",t_stop= 7.034e-17, t=0.0, degree=1)
## bc - P (imposed Pressure)
Gamma_P_0 = DirichletBC(V_p, P_R, boundaries, 1)
Gamma_P_1 = DirichletBC(V_p, P_L, boundaries, 2)
Gamma_P_2 = DirichletBC(V_p, P_T, boundaries, 3)
Gamma_P_3 = DirichletBC(V_p, P_B, boundaries, 4)
Gamma_P_4 = DirichletBC(V_p, P_V, boundaries, 5)
bc_P = [Gamma_P_0,Gamma_P_1,Gamma_P_2,Gamma_P_3,Gamma_P_4]

###############################################################
Ini_P=Constant(1e6) #P0=0.1e6 Pa
P_prev = interpolate(Expression('Ini_P', Ini_P=1e6, degree=1), V_p)

Porous = pore/N*p_*p_t*dx + dt*(kappa/mu_dynamic)*rho*inner(nabla_grad(p_), nabla_grad(p_t))*dx-(pore/N*P_prev +dt*f)*p_t*dx
#+kappa* dot(P_, Flux_T)*ds(3) +kappa* dot(P_, Flux_B) *ds(4) + kappa*dot(Flux_C, P_)* ds(5) 

#################################################################
maxdiff=1.
t = dt
toll=1.e-10
err_P = 1
    #solver_T.solve()
while err_P>toll and t <= t_stop:
    P_V.t = t

    plot(p_, interactive=False)
    solve(Porous == 0, p_, bc_P,
      solver_parameters={"newton_solver":{"relative_tolerance":1e-10}})

    rho=EqOfState(p_.vector().array())
    N=EqOfState_N(rho)

    plot(p_, interactive=False)

    maxdiff = np.abs(p_.vector().array()-P_prev.vector().array()).max()
    print 'Max error, t=%.2f: %-10.3f' % (t, maxdiff)

    # visualization statements
    t += dt
    P_prev.assign(p_)


# Save solution in VTK format
file_P = File(save_dir+"/p_.pvd") # use .pvd if .xdmf is not working
file_P << p_

# Plot solution
plot(p_, interactive=True)
