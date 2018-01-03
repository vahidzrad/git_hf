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
from gradient_damage import *
from fracking_basic import *
from CrkOpn_MosVah import Opening
from Sneddon import SneddonWidth
import os
import matplotlib.pyplot as plt

P_constant_list=[0.1]
hsize_list = [0.01 ]
colors_i = ['r', 'b', 'g','m','c','k']



for P_constant in P_constant_list:
	for hsize in hsize_list:
		ell_list = [1*hsize, 2*hsize, 3*hsize]
		fig = plt.figure()
		for (i, ell) in enumerate(ell_list):
		    	# Varying the hsize mesh size
		       	problem = Fracking(hsize, ell, P_constant)
			problem.solve()

			arr_Coor_plt_X, arr_li = Opening(hsize)
			x, x_, uy, uy_ = SneddonWidth(P_constant)

			plt.plot(arr_Coor_plt_X, arr_li, color = colors_i[i], label="$\ell=$ %g, $p=$ %g"%(round(ell,4),P_constant))

		plt.plot(x, uy, '-', dashes=[8, 4, 2, 4, 2, 4], color = colors_i[i+1], label='Senddon, $p=$ %s'%P_constant)
		plt.plot(x_, uy_, '-', dashes=[8, 4, 2, 4, 2, 4], color = colors_i[i+1])

		plt.xlabel("Fracture Length")
		plt.ylabel("Width")

		fig.suptitle("$h$= %g, $p=$%g"%(round(hsize,2), P_constant) , fontsize=14)

		plt.legend(loc="best")
		save_fig = "hydraulicfracturing-results/"
		plt.savefig(save_fig +"h= %g, p=%g"%(round(hsize,2), P_constant)+".pdf")



#### Remove the .pyc file ####
MPI.barrier(mpi_comm_world())
if MPI.rank(mpi_comm_world()) == 0:
    os.remove("fracking_basic.pyc")