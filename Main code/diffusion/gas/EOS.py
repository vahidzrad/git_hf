from dolfin import *
from mshr import *
import sys, os, sympy, shutil, math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
from numpy import loadtxt 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def EqOfState(pressu):
	pressu=np.array(pressu)*1e6#*100 #centipascal, #1 pascal is equal to 10 decipascal, or 100 centipascal

	R = 0.1889241*1000 #J/(kgK)
	T_c = 304.1282 #K
	rho_c =467.6 #kg/m^3  
	#p=(1+delta*phi_r_d)*rho*R*T---->dimension: Joule=N.m-->p: Pa
				   		############################################################################################### 
	ni = loadtxt("GasFluidFlow_input/ni.txt", comments="#", delimiter=",", unpack=False)
	di = loadtxt("GasFluidFlow_input/di.txt", comments="#", delimiter=",", unpack=False)
	ai = loadtxt("GasFluidFlow_input/ai.txt", comments="#", delimiter=",", unpack=False)
	ti = loadtxt("GasFluidFlow_input/ti.txt", comments="#", delimiter=",", unpack=False)
	bi = loadtxt("GasFluidFlow_input/bi.txt", comments="#", delimiter=",", unpack=False)
	ci = loadtxt("GasFluidFlow_input/ci.txt", comments="#", delimiter=",", unpack=False)
	alpha_i = loadtxt("GasFluidFlow_input/alphai.txt", comments="#", delimiter=",", unpack=False)
	beta_i = loadtxt("GasFluidFlow_input/betai.txt", comments="#", delimiter=",", unpack=False)
	gamma_i = loadtxt("GasFluidFlow_input/gammai.txt", comments="#", delimiter=",", unpack=False)
	epsilon_i = loadtxt("GasFluidFlow_input/epsiloni.txt", comments="#", delimiter=",", unpack=False)
	A_i = loadtxt("GasFluidFlow_input/A_i.txt", comments="#", delimiter=",", unpack=False)
	B_i = loadtxt("GasFluidFlow_input/B_i.txt", comments="#", delimiter=",", unpack=False)
	C_i = loadtxt("GasFluidFlow_input/C_i.txt", comments="#", delimiter=",", unpack=False)
	D_i = loadtxt("GasFluidFlow_input/D_i.txt", comments="#", delimiter=",", unpack=False)

	#############################################################################################
	load_steps=1000
	temprature=[25+273.15, 45+273.15,65+273.15] # Celsius+273.15=Kelvin
	rho_vector = np.linspace(0.01, 10000, load_steps) #kg/m^3

	i_t=len(temprature)
	i_b=len(rho_vector)
	pressure = np.zeros((len(temprature),len(rho_vector)+1))
	N = np.zeros((len(temprature),len(rho_vector)+1))
	for i in range(i_t):
		T=temprature[i]
		for j in range(i_b):
			rho=rho_vector[j]
			delta=rho/rho_c
			tau=T_c/T

			#print di[0:7]
			#print di[7:39]
			#print di[39:42]
			#phi1-7############################################################################################

			phi1_7      = sum(ni[0:7]*delta**di[0:7]*tau**ti[0:7])
			phi_d_1_7   = sum(ni[0:7]*di[0:7]*delta**(di[0:7]-1.)*tau**ti[0:7])
			phi_d_d_1_7 = sum(ni[0:7]*di[0:7]*(di[0:7]-1.)*delta**(di[0:7]-2.)*tau**ti[0:7])
			#phi8-34############################################################################################

			phi8_34		 = sum(ni[7:34] * delta**di[7:34] * tau**ti[7:34] * np.exp(-delta**ci[7:34]))
			phi_d_8_34 	 = sum(ni[7:34] * np.exp(-delta**ci[7:34]) * (delta**(di[7:34]-1) * tau**ti[7:34]*(di[7:34]-ci[7:34]* delta** ci[7:34])) )
			phi_d_d_8_34	 = sum(ni[7:34] * np.exp(-delta**ci[7:34]) * (delta**(di[7:34]-2) * tau**ti[7:34]*((di[7:34]-ci[7:34]* delta** ci[7:34])*(di[7:34]-1-ci[7:34]* delta** ci[7:34])-ci[7:34]**2 * delta** ci[7:34]) ))
			#phi35-39############################################################################################
			phi35_39 =sum(ni[34:39] * delta**di[34:39] * tau**ti[34:39] * np.exp(-alpha_i[34:39]*(delta-epsilon_i[34:39])**2-beta_i[34:39]*(tau-gamma_i[34:39])**2))

	   		phi_d_35_39 =sum( ni[34:39] * delta**di[34:39] * tau**ti[34:39] * np.exp(-alpha_i[34:39]*(delta-epsilon_i[34:39])**2-beta_i[34:39]*(tau-gamma_i[34:39])**2) *(di[34:39]/delta-2*alpha_i[34:39]*(delta-epsilon_i[34:39])) )

			phi_d_d_35_39 =sum(ni[34:39] * tau**ti[34:39] * np.exp(-alpha_i[34:39]*(delta-epsilon_i[34:39])**2-beta_i[34:39]*(tau-gamma_i[34:39])**2) *(-2*alpha_i[34:39]*delta**di[34:39] + 4* alpha_i[34:39]**2 * delta**di[34:39] *(delta-epsilon_i[34:39])**2 - 4* di[34:39]* alpha_i[34:39] *  delta**(di[34:39]-1) *(delta-epsilon_i[34:39])+(di[34:39])*(di[34:39]-1)*delta**(di[34:39]-2)))
			#############################################################################################
			thetaa=sum((1-tau)+A_i[39:42]*((delta-1)**2)**(1/(2*beta_i[39:42])))
			Delta= sum(thetaa**2+B_i[39:42]*((delta-1)**2)**ai[39:42])
			Si=sum(np.exp(-C_i[39:42]*(delta-1)**2 - D_i[39:42]*(tau-1)**2))

			d_Delta_d_delt=sum((delta-1)*(A_i[39:42]*thetaa*2/beta_i[39:42]*((delta-1)**2)**(1/(2*beta_i[39:42])-1))+2*B_i[39:42]*ai[39:42]*((delta-1)**2)**(ai[39:42]-1))

			d2_Delta_d_delt=sum(1/(delta-1)*d_Delta_d_delt + (delta-1)**2 * (4*B_i[39:42]*ai[39:42]*(ai[39:42]-1)*((delta-1)**2)**(ai[39:42]-2)+2*(A_i[39:42])**2 * (1/beta_i[39:42])**2 * ( ((delta-1)**2)**(1/(2*beta_i[39:42])-1) )**2+A_i[39:42]*thetaa*4/beta_i[39:42]*(1/(2*beta_i[39:42])-1) * ( ((delta-1)**2)**(1/(2*beta_i[39:42])-2))))


			#phi40-42############################################################################################
			phi40_42= sum(ni[39:42] * Delta**bi[39:42] * delta * Si)
			phi_d_40_42=sum(ni[39:42] * (Delta**bi[39:42] * (Si + delta*(-2*C_i[39:42]*(delta-1)*Si))+(bi[39:42]* Delta**(bi[39:42]-1))*d_Delta_d_delt*(delta * Si)))
			phi_d_d_40_42=sum(ni[39:42] * (Delta**bi[39:42]*(2*(-2*C_i[39:42]*(delta-1)*Si)+delta*( (2*C_i[39:42]*(delta-1)**2-1)*2*C_i[39:42]*Si))+2* d_Delta_d_delt*(Si+delta*(-2*C_i[39:42]*(delta-1)*Si))+ d2_Delta_d_delt*delta*Si) )


			#############################################################################################
			phi_r=phi1_7+phi8_34+phi35_39+phi40_42
			phi_r_delta=phi_d_1_7+phi_d_8_34+phi_d_35_39+phi_d_40_42
			phi_r_delta_delta = phi_d_d_1_7 + phi_d_d_8_34 + phi_d_d_35_39 + phi_d_d_40_42
			#############################################################################################
			#print phi_r_delta
			#print (1+phi_r_delta)*(rho*R*T)
			pressure[i,j+1]=(1+delta*phi_r_delta)*(rho*R*T) # Pa=N/m^2
			#print pressure

			N[i,j+1]=R*T*(delta**2 * phi_r_delta_delta +2*delta*phi_r_delta + 1) # J/(Kg*K)*K-->J/9.806N-->1/9.806 m



	#plt.plot(pressure[0,:]/1e6, np.r_[0, rho_vector],'r--' ,label='T=25')
	#plt.plot(pressure[1,:]/1e6, np.r_[0, rho_vector],'bs--',label='T=45')
	#plt.plot(pressure[2,:]/1e6, np.r_[0, rho_vector],'g^--',label='T=65')
	#plt.xlabel('Pressure (MPa)')
	#plt.ylabel('Density (kg/m^3)')
	#plt.axis([0, 50, 0, 900])
	#plt.legend( loc='lower right', numpoints = 1 )

	#plt.show()
        #plt.interactive(False)

	x = pressure[2,:] # 	Pressure with temprature 45
	y = np.r_[0, rho_vector]



	f_rho = interp1d(x, y)
	f2 = interp1d(x, y, kind='cubic')
	#xnew = np.linspace(0, 10, num=13, endpoint=True)

	return f_rho(pressu) 

if __name__ == '__main__':
  	 	 # test1.py executed as script
  	 	 # do something
  	 	 some_func()



