# Copyright (C) 2014 Tianyi Li
# Licensed under the GNU LGPL Version 3.

from dolfin import *
from mshr import *
import sys, os, sympy, shutil, math
import numpy as np
# Problem
class Fracking():
	    # ==========================================================================================
	    # Loading
	    # ==========================================================================================
	    ut =1.e-2 # reference value for the loading (imposed displacement)
	    f = 0. # bulk load
	    load_min = 0. # load multiplier min value

	    load_max = 1. # load multiplier max value
	    load_steps = 100 # number of time steps
	    # To define an initial pressure field
	    p_max = 0.5e0
	    # ==========================================================================================
	    # Numerical parameters of the alternate minimization
	    # ==========================================================================================
	    maxiter = 1000 
	    toll = 1e-5
	    # ==========================================================================================
	    # Material constant and propertices
	    # ==========================================================================================
	    law = "AT1"
	    #k = Constant()  # k parameter for the "ATk" model
            kres = Constant(1e-6)
	    E = Constant(1.)
	    nu = Constant(0.)
	    Gc = Constant(1.)
	    mu = E/(2.0*(1.0+ nu))
	    kappa=E/(2*(1.0 + nu)*(1. - 2.0 * nu))
	    C_biot= Constant(0) #Added by Mostafa
	    kres= 1e-6
	    pstress=False


   	    def __init__(self, hsize, ell):
 	        # Parameters
  	      	self.hsize = hsize
   	    	self.ell = ell


	    def prefix(self):
	        return "hydraulicfracturing"


	    def	SayHello(self):
		print "hsize=", self.hsize
		print "ell=", self.ell
		print "ut=", self.ut



    def a(self, alpha):
        """
        Modulation of the elastic stiffness
        """
        if self.law == "AT1":
            return (1-alpha)**2
        elif self.law == "AT2":
            return (1-alpha)**2
        elif self.law == "ATk":
            return (1-self.w(alpha))/(1+(self.k-1)*self.w(alpha))

    def w(self, alpha):
        """
        Local energy dissipation
        """
        if self.law == "AT1":
            return alpha
        elif self.law == "AT2":
            return alpha**2
        elif self.law == "ATk":
            return 1-(1-alpha)**2

    def eps(self, u):
        """
        Geometrical strain
        """
        return sym(grad(u))

    def eps0(self):
        """
        Inelastic strain, supposed to be isotropic
        """
        return 0.

    def epse(self, u):
        """
        Elastic strain
        """
        Id = Identity(len(u))
        return self.eps(u) - self._eps0*Id

    def sigma0(self, u):
        """
        Application of the sound elasticy tensor on the strain tensor
        """
        Id = Identity(len(u))
        return 2.0*self.mu*self.epse(u) + self.lmbda*tr(self.epse(u))*Id


    def sigma(self, u, alpha):
        """
        Stress
        """
        return (self.a(alpha)+self.kres) * self.sigma0(u)

    def elastic_energy_density(self, u, alpha):
        """
        Elastic energy density
        """
        return 0.5 * inner(self.sigma(u, alpha), self.epse(u))

    def dissipated_energy_density(self, u, alpha):
        """
        Dissipated energy density
        """
        z = sympy.Symbol("z", positive=True)
        self.c_w = float(4*sympy.integrate(sympy.sqrt(self.w(z)), (z, 0, 1)))
        return self.Gc/self.c_w * (self.w(alpha)/self.ell + self.ell*inner(grad(alpha), grad(alpha)))




		
if __name__ == '__main__':

    # Run a fast simulation
    problem = Fracking(hsize=0.1, ell=1.0e-2)
    problem.SayHello()
	




