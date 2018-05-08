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

from Fracking_tao_basic import Fracking
from CrkOpn_MosVah import Opening
from Sneddon import SneddonWidth
import os
import matplotlib.pyplot as plt

pressure_max_list=[1]
hsize_list = [0.04, 0.02, 0.01,0.005, 0.0025 ]
colors_i = ['r', 'b', 'g','m','c','k']

E = 1. # Young modulus
nu = 0.0 # Poisson ratio

ModelB= False 
if not ModelB:  # Model A (isotropic model)
	Model = 'Isotropic'
else:  # Model B (Amor's model)
	Model = 'Amor'

law='AT1'

Volume_num  =np.zeros((max(len(pressure_max_list),len(hsize_list)), 4)) #4 is thelength of ell_list

for (k, pressure_max) in enumerate(pressure_max_list):
	fig = plt.figure()
	for (j, hsize) in enumerate(hsize_list):
		#ell_list = [100*hsize, 150*hsize, 200*hsize]
		ell_list = [0.24]





		for (i, ell) in enumerate(ell_list):
		    	# Varying the hsize mesh size
		       	Fracking(hsize, pressure_max, ell,E, nu, Model, law)
			arr_Coor_plt_X, arr_li, Volume = Opening(hsize,ell, Model, law)

			print "Numeric Volume=%g"%Volume
			Volume_num[j][i]=Volume

			plt.plot(arr_Coor_plt_X, arr_li, label="$\ell=$ %g, $h=$ %g"%(round(ell,4), round(hsize,4)))

	x, x_, width, width_, volumeAnalytical = SneddonWidth(pressure_max,E, nu) 
	plt.plot(x, width, '-', dashes=[8, 4, 2, 4, 2, 4], color = colors_i[j])
	plt.plot(x_, width_, '-', dashes=[8, 4, 2, 4, 2, 4], color = colors_i[j])

	plt.xlabel("Fracture Length")
	plt.ylabel("Width")

	fig.suptitle("$%s, p=$%g"%(law, pressure_max) , fontsize=14)

	plt.legend(loc="best")
	save_fig = "Fracking_result/"
	plt.savefig(save_fig +"$%s, p=$%g"%(law, pressure_max)+".pdf")

print "Numeric Volume=",Volume_num
print "Analytical Volume=",volumeAnalytical





#### Remove the .pyc file ####
MPI.barrier(mpi_comm_world())
if MPI.rank(mpi_comm_world()) == 0:
    os.remove("Fracking_tao_basic.pyc")
    os.remove("CrkOpn_MosVah.pyc")
    os.remove("Sneddon.pyc")
