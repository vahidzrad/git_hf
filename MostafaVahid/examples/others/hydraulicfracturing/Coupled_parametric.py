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
from Diffusion_gas import Diffusion
from Sneddon import SneddonWidth
import os
import matplotlib.pyplot as plt

P_constant= 0.1
hsize = 0.01 
ell = 0.01





problem = Fracking(hsize, ell, P_constant)
problem.solve()

#Diffusion()

#### Remove the .pyc file ####
MPI.barrier(mpi_comm_world())
if MPI.rank(mpi_comm_world()) == 0:
    os.remove("fracking_basic.pyc")
