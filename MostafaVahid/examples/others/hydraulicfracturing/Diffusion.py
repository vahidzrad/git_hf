
from dolfin import *
from mshr import *
import sys, os, sympy, shutil, math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D                                          
import matplotlib.pyplot as plt 

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
savedir = "results/bar"
#mesh = UnitSquareMesh(64, 64)
mesh = Mesh('Ntch_edge_mesh/General.xml')
V = FunctionSpace(mesh, 'Lagrange', 1)
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
L=4.
H=4.
T_A = 10e6; omega = 2*pi
T_0 = Expression('T_A', T_A=T_A)

# -----------------------------------------------------------------------------------------
# Define boundary sets for boundary conditions
#------------------------------------------------------------------------------------------

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


class CenterCrk(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H/2, eps=DOLFIN_EPS) and x[0] >= 1.8 and x[0] <= 2.2 

class Void(SubDomain):
    def inside(self, x, on_boundary):
        return   x[0] <= 2.5 and x[0] >= 1.5 and  x[1] <= 2.5 and x[1] >= 1.5 and on_boundary
       
# Initialize sub-domain instances
right=Right()
left=Left()
top=Top()
bottom=Bottom()
centerCrk=CenterCrk()
void=Void()
# define meshfunction to identify boundaries by numbers
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(9999)
right.mark(boundaries, 1) # mark top as 1
left.mark(boundaries, 2) # mark top as 2
top.mark(boundaries, 3) # mark top as 3
bottom.mark(boundaries, 4) # mark bottom as 4
centerCrk.mark(boundaries, 5) # mark center as 5
void.mark(boundaries, 6) 

# Define new measure including boundary naming 
ds = Measure("ds")[boundaries] # left: ds(1), right: ds(2)
#------------------------------------------------------------------------------------------
# Define boundary condition
T_R = Constant(10e6)
T_L = Constant(10e6)
T_T = Constant(20e6)
T_B = Constant(10e6)
T_C = Constant(25e6)

## bc - u (imposed temprature)
Gamma_T_0 = DirichletBC(V, T_R, boundaries, 1)
Gamma_T_1 = DirichletBC(V, T_L, boundaries, 2)
Gamma_T_2 = DirichletBC(V, T_T, boundaries, 3)
Gamma_T_3 = DirichletBC(V, T_B, boundaries, 4)
Gamma_T_4 = DirichletBC(V, T_C, boundaries, 6)
bc_T = [Gamma_T_0,Gamma_T_1,Gamma_T_2,Gamma_T_3]#,Gamma_T_4]



T_prev = interpolate(Constant(T_R), V)


nIter=100
M_biot=1.
kappa=1e10

t_stop = H/kappa  # 20 time steps per period
dt=t_stop/nIter


T = Function(V)
v = TestFunction(V)

f = Constant(0)



Heat = M_biot*T*v*dx +dt*kappa*inner(nabla_grad(T), nabla_grad(v))*dx-(M_biot*T_prev +dt*f)*v*dx#+kappa* dot(P_, Flux_T)*ds(3) +kappa* dot(P_, Flux_B) *ds(4) + kappa*dot(Flux_C, P_)* ds(5) 

#Heat = rho*c*T*v*dx +\
 #dt*kappa*inner(nabla_grad(T), nabla_grad(v))*dx-\
#(rho*c*T_prev + dt*f)*v*dx +\
#kappa* dot(v, Flux_T)*ds(3) +\
#kappa* dot(v, Flux_B) *ds(4) + kappa*dot(Flux_C, v)* ds(5) 


maxdiff=1.
t = dt
toll=1.e-5
err_T = 1

while err_T>toll and t <= t_stop:
    T_0.t = t
    #solver_T.solve()

    solve(Heat == 0, T, bc_T,
      solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})

    maxdiff = np.abs(T.vector().array()-T_prev.vector().array()).max()
    print 'Max error, t=%.2f: %-10.3f' % (t, maxdiff)

    # visualization statements
    t += dt
    T_prev.assign(T)


# Save solution in VTK format
file_T = File(savedir+"/T.pvd") # use .pvd if .xdmf is not working
file_T << T

# Plot solution
plot(T, interactive=True)


