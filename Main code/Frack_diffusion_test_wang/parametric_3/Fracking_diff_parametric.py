# Copyright (C) 2017 Corrado Maurini, Tianyi Li
#
# This file is part of FEniCS Gradient Damage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
from dolfin import *
from mshr import *
import sys, os, sympy, shutil, math
import numpy as np
import matplotlib.pyplot as plt

from Fracking_tao_diffusion import Fracking





hsize =  1.5

colors_i = ['r', 'b', 'g','m','c','k']


E = 6.e3 # Young modulus
nu = 0.34 # Poisson ratio


#K=1.5e3
#G= 1.5e3

#E=(9*K*G)/(3*K+G)
#nu=(3*K-2*G)/(2*(3*K+G))

ModelB= False 
law='AT1'

ka_list = [7.]
kb = 1.
q =9.e-2*0.4 #6.e-5

mu_dynamic_list= [1.e-11, 1.e-10,1.e-9]

for (k, mu_dynamic) in enumerate(mu_dynamic_list):
	for (j, ka) in enumerate(ka_list):
		ell_list = [2*hsize]

		for (i, ell) in enumerate(ell_list):
		    	# Varying the hsize mesh size
		       	Fracking(E, nu, hsize, ell, law, ModelB, ka, kb, k, mu_dynamic)








#### Remove the .pyc file ####
MPI.barrier(mpi_comm_world())
if MPI.rank(mpi_comm_world()) == 0:
    os.remove("Fracking_tao_diffusion.pyc")

