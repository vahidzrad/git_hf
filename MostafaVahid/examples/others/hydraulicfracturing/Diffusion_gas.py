
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


def Diffusion():

	hsize = 0.1 
	#-----------------------------------------------------------------------------------------
	#-----------------------------------------------------------------------------------------
	getcontext().prec = 100
	#----------------------------------------------------------------------------------------
	#----------------------------------------------------------------------------------------
	savedir = "Pressure_results"
	mesh = Mesh("meshes/fracking_hsize"+str(float(hsize))+".xml")
	L=4.
	H=4.
	#----------------------------------------------------------------------------------------
	#----------------------------------------------------------------------------------------
	V = FunctionSpace(mesh, 'Lagrange', 1)
	P = Function(V)   #pressure Pa
	P_ = TestFunction(V)
	rho = Function(V)  #density kg/m^3
	N = Function(V)	   #N=R*T*(delta**2 * phi_r_delta_delta +2*delta*phi_r_delta + 1) # J/(Kg*K)*K-->m^2/s^2
	#-----------------------------------------------------------------------------------------
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


	class Void(SubDomain):
	    def inside(self, x, on_boundary):
		return   x[0] <= 2.5 and x[0] >= 1.5 and  x[1] <= 2.5 and x[1] >= 1.5 and on_boundary
	       
	# Initialize sub-domain instances
	right=Right()
	left=Left()
	top=Top()
	bottom=Bottom()
	void=Void()
	# define meshfunction to identify boundaries by numbers
	boundaries = FacetFunction("size_t", mesh)
	boundaries.set_all(9999)
	right.mark(boundaries, 1) # mark right as 1
	left.mark(boundaries, 2) # mark left as 2
	top.mark(boundaries, 3) # mark top as 3
	bottom.mark(boundaries, 4) # mark bottom as 4
	void.mark(boundaries, 5) # mark void as 5

	# Define new measure including boundary naming 
	ds = Measure("ds")[boundaries] # left: ds(1), right: ds(2)
	#------------------------------------------------------------------------------------------
	#------------------------------------------------------------------------------------------
	pore=0.01 #porosity
	mu_dynamic=4.4e-5 #dynamic viscosity, Pa.s
	kappa=1e+10 #permeability, m^2
	Flux_T=Constant(0.)
	Flux_B=Constant(0.)
	Flux_C=Constant(0.)
	f = Constant(0)

	Rho_a=EqOfState(0.01) #initial density according to initial pressure(Pa?)
	N_a=EqOfState_N([1.565353012e-10]) #initial N(m^2/s^2) according to initial density(kg/m^3)
	print "Rho_intial=",Rho_a
	print "N_intial=",N_a[0]


	#------------------------------------------------------------------------------------------
	nIter=20 # number of time steps
	t_stop = H**2 *mu_dynamic/(kappa*Rho_a*N_a[0])  # non dimensional time (s) 
	dt=t_stop/nIter 
	print  "t_stop=",t_stop

	Rho_a=Constant(Rho_a)
	t_stop_a=Constant(t_stop)
	N_a=Constant(N_a[0])
	#------------------------------------------------------------------------------------------
	rho = interpolate(Expression('Rho', Rho = Rho_a, degree=1), V)
	N = interpolate(Expression('N', N= N_a, degree=1), V)
	#------------------------------------------------------------------------------------------
	#------------------------------------------------------------------------------------------
	# Define boundary condition
	P_R = Constant(0.01) #0.01 =1MPa
	P_L = Constant(0.01)
	P_B = Constant(0.01)
	P_T = Constant(0.01)
	P_V = Expression("t/t_stop*0.02",t_stop= t_stop_a, t=0.0, degree=1)
	## bc - P (imposed Pressure)
	Gamma_P_0 = DirichletBC(V, P_R, boundaries, 1)
	Gamma_P_1 = DirichletBC(V, P_L, boundaries, 2)
	Gamma_P_2 = DirichletBC(V, P_T, boundaries, 3)
	Gamma_P_3 = DirichletBC(V, P_B, boundaries, 4)
	Gamma_P_4 = DirichletBC(V, P_V, boundaries, 5)
	bc_P = [Gamma_P_0,Gamma_P_1,Gamma_P_2,Gamma_P_3,Gamma_P_4]

	###############################################################
	Ini_P_a=Constant(0.01) #P0=0.1e6 Pa
	P_prev = interpolate(Expression('Ini_P', Ini_P=Ini_P_a, degree=1), V)

	Porous = pore/N*P*P_*dx + dt*(kappa/mu_dynamic)*rho*inner(nabla_grad(P), nabla_grad(P_))*dx-(pore/N*P_prev +dt*f)*P_*dx
	#+kappa* dot(P_, Flux_T)*ds(3) +kappa* dot(P_, Flux_B) *ds(4) + kappa*dot(Flux_C, P_)* ds(5) 

	#################################################################
	maxdiff=1.
	t = dt
	toll=1.e-10
	err_P = 1
	    #solver_T.solve()
	while err_P>toll and t <= t_stop:
	    P_V.t = t

	    plot(P, interactive=False)
	    solve(Porous == 0, P, bc_P,
	      solver_parameters={"newton_solver":{"relative_tolerance":1e-10}})

	    rho=EqOfState(P.vector().array())
	    N=EqOfState_N(rho)

	    plot(P, interactive=False)

	    maxdiff = np.abs(P.vector().array()-P_prev.vector().array()).max()
	    print 'Max error, t=%.2f: %-10.3f' % (t, maxdiff)

	    # visualization statements
	    t += dt
	    P_prev.assign(P)


	# Save solution in VTK format
	file_P = File(savedir+"/P.pvd") # use .pvd if .xdmf is not working
	file_P << P

	# Plot solution
	plot(P, interactive=True)
	
	output_file_pressure = HDF5File(mpi_comm_world(), "pressure.h5", "w") # self.save_dir + "uO.h5"
	output_file_pressure.write(P, "solution")
	output_file_pressure.close()
	


if __name__ == '__main__':
  	 	 # test1.py executed as script
  	 	 # do something
  	 	 Diffusion()
