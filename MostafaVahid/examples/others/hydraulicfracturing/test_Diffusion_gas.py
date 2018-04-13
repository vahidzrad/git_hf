
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
	#-----------------------------------------------------------------------------------------
	#Input data:
	#-----------------------------------------------------------------------------------------
	nIter=2 # number of time steps
	hsize = 0.01 # mesh size for reading the mesh file generated by phase field class
	pore=0.01 #porosity
	mu_dynamic=4.4e-5 #dynamic viscosity, Pa.s
	kappa=1e+10 #permeability, m^2
	Flux_T=Constant(0.)
	Flux_B=Constant(0.)
	Flux_C=Constant(0.)
	f = Constant(0)
	getcontext().prec = 100
	savedir = "Pressure_results"
	mesh = Mesh("meshes/fracking_hsize"+str(float(hsize))+".xml")
	#mesh_fun = MeshFunction("size_t", mesh,"meshes/fracking_hsize"+str(float(hsize))+"_facet_region.xml")



	L=4.
	H=4.
	Init_Pressure=0.15
	#----------------------------------------------------------------------------------------
	# Define spaces
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

	class Circle(SubDomain):
	    def inside(self, x, on_boundary):
		return  sqrt( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) - 0.25 < 1e-6  and on_boundary

        class Crack(SubDomain):
            def inside(self, x, on_boundary):
  		return near((x[1] - 2.) * 0.01, 0) and  x[0] <= 2.2 and x[0] >= 1.8 and on_boundary


	       
	# Initialize sub-domain instances
	right=Right()
	left=Left()
	top=Top()
	bottom=Bottom()
	void=Void()
	circle=Circle()
	crack=Crack()


	# define meshfunction to identify boundaries by numbers
	boundaries = FacetFunction("size_t", mesh)
	boundaries.set_all(9999)
	right.mark(boundaries, 1) # mark right as 1
	left.mark(boundaries, 2) # mark left as 2
	top.mark(boundaries, 3) # mark top as 3
	bottom.mark(boundaries, 4) # mark bottom as 4
	void.mark(boundaries, 5) # mark void as 5
	circle.mark(boundaries, 6) # mark circle as 6
	crack.mark(boundaries, 7) # mark crack as 7



	# Define new measure including boundary naming 
	ds = Measure("ds")[boundaries] # left: ds(1), right: ds(2)
	#------------------------------------------------------------------------------------------
	# Define initial time, density and N
	#------------------------------------------------------------------------------------------
	Rho_intial=EqOfState(Init_Pressure) #initial density according to initial pressure(Pa?)
	N_intial=EqOfState_N([Rho_intial]) #initial N(m^2/s^2) according to initial density(kg/m^3)
	#-----------------------
	t_stop = H**2 *mu_dynamic/(kappa*Rho_intial*N_intial[0])  # non dimensional time (s) 
	dt=t_stop/nIter 

	print "Rho_intial=",Rho_intial
	print "N_intial=",N_intial[0]
	print  "t_stop=",t_stop
	#------------------------------------------------------------------------------------------
	N_intial=Constant(N_intial[0])
	t_stop_a=Constant(t_stop)
	Rho_intial=Constant(Rho_intial)

	rho = interpolate(Expression('Rho', Rho = Rho_intial, degree=1), V)
	N = interpolate(Expression('N', N= N_intial, degree=1), V)
	#------------------------------------------------------------------------------------------
	# Define boundary condition
	#------------------------------------------------------------------------------------------
	P_R = Constant(Init_Pressure) #0.01 =1MPa
	P_L = Constant(Init_Pressure)
	P_B = Constant(Init_Pressure)
	P_T = Constant(Init_Pressure)
	#P_V = Expression("100*t/t_stop*Init_Pressure",t_stop= t_stop_a, Init_Pressure=Init_Pressure, t=0.0, degree=1)
	P_V = Constant(10*Init_Pressure)
	## bc - P (imposed Pressure)
	Gamma_P_0 = DirichletBC(V, P_R, boundaries, 1)
	Gamma_P_1 = DirichletBC(V, P_L, boundaries, 2)
	Gamma_P_2 = DirichletBC(V, P_T, boundaries, 3)
	Gamma_P_3 = DirichletBC(V, P_B, boundaries, 4)
	Gamma_P_4 = DirichletBC(V, P_V, boundaries, 7)
	#Gamma_P_4 = DirichletBC(V, P_V, mesh_fun, 1)

	bc_P = [Gamma_P_0,Gamma_P_1,Gamma_P_2,Gamma_P_3 ,Gamma_P_4]

	###############################################################
	P_prev = interpolate(Expression('Ini_P', Ini_P=Init_Pressure, degree=1), V)
	Porous = pore/N*P*P_*dx + dt*(kappa/mu_dynamic)*rho*inner(nabla_grad(P), nabla_grad(P_))*dx-(pore/N*P_prev +dt*f)*P_*dx
	#+kappa* dot(P_, Flux_T)*ds(3) +kappa* dot(P_, Flux_B) *ds(4) + kappa*dot(Flux_C, P_)* ds(5) 
	#################################################################
	maxdiff=1.
	t = dt
	toll=1.e-10
	err_P = 1

	while err_P>toll and t <= t_stop:
	    P_V.t = t

	    plot(P, interactive=False)
	    solve(Porous == 0, P, bc_P,
	      solver_parameters={"newton_solver":{"relative_tolerance":1e-10}})

	    rho=EqOfState(P.vector().array()) 	#set the new density according new pressure
	    N=EqOfState_N(rho)			#set the new N according new pressure

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