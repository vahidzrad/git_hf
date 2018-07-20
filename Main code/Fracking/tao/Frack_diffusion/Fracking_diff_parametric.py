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





hsize =  0.75

colors_i = ['r', 'b', 'g','m','c','k']


E = 6.e3 # Young modulus
nu = 0.34 # Poisson ratio

ModelB= False 
law='AT1'

ka_list = [7.]
kb = 1.
q =6.e-5

pressure_steps_list= [10]

for (k, pressure_steps) in enumerate(pressure_steps_list):
	fig = plt.figure()
	for (j, ka) in enumerate(ka_list):
		ell_list = [2*hsize]

		for (i, ell) in enumerate(ell_list):
		    	# Varying the hsize mesh size
		       	Fracking(E, nu, hsize, ell, law, ModelB, ka, kb, q, pressure_steps)








#### Remove the .pyc file ####
MPI.barrier(mpi_comm_world())
if MPI.rank(mpi_comm_world()) == 0:
    os.remove("Fracking_tao_diffusion.pyc")

