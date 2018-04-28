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

from Fracking_SNES_basic import Fracking
from CrkOpn_MosVah import Opening
from Sneddon import SneddonWidth
import os
import matplotlib.pyplot as plt

pressure_max_list=[ 0.1]
hsize_list = [0.001 ]
colors_i = ['r', 'b', 'g','m','c','k']

Volume_num=np.zeros((len(pressure_max_list), 4)) #4 is thelength of ell_list
Volume_analy=np.zeros((len(pressure_max_list), 4)) #4 is thelength of ell_list

for (k, pressure_max) in enumerate(pressure_max_list):
	for (j, hsize) in enumerate(hsize_list):
		ell_list = [1*hsize, 2*hsize, 3*hsize, 4*hsize]

		#Volume_num=np.zeros((len(hsize_list), len(ell_list)))
		#Volume_analy=np.zeros((len(hsize_list), len(ell_list)))


		fig = plt.figure()
		for (i, ell) in enumerate(ell_list):
		    	# Varying the hsize mesh size
		       	Fracking(hsize, pressure_max, ell)
			

			arr_Coor_plt_X, arr_li, Volume = Opening(hsize,ell)
			x, x_, width, width_, volumeAnalytical = SneddonWidth(pressure_max) 

			print "Numeric Volume=%g"%Volume
			print "Analytical Volume=%g"%volumeAnalytical
			#Volume_num[j][i]=Volume
			#Volume_analy[j][i]=volumeAnalytical
			Volume_num[k][i]=Volume
			Volume_analy[k][i]=volumeAnalytical

			plt.plot(arr_Coor_plt_X, arr_li, label="$\ell=$ %g, $p=$ %g"%(round(ell,4),pressure_max))

		plt.plot(x, width, '-', dashes=[8, 4, 2, 4, 2, 4], color = colors_i[j], label='Senddon, $p=$ %s'%pressure_max)
		plt.plot(x_, width_, '-', dashes=[8, 4, 2, 4, 2, 4], color = colors_i[j])

		plt.xlabel("Fracture Length")
		plt.ylabel("Width")

		fig.suptitle("$h$= %g, $p=$%g"%(round(hsize,2), pressure_max) , fontsize=14)

		plt.legend(loc="best")
		save_fig = "Fracking_result/"
		plt.savefig(save_fig +"h= %g, p=%g"%(round(hsize,2), pressure_max)+".pdf")
		print "Numeric Volume=",Volume_num
		print "Analytical Volume=",Volume_analy





#### Remove the .pyc file ####
MPI.barrier(mpi_comm_world())
if MPI.rank(mpi_comm_world()) == 0:
    os.remove("Fracking_SNES_basic.pyc")
    os.remove("CrkOpn_MosVah.pyc")
    os.remove("Sneddon.pyc")
