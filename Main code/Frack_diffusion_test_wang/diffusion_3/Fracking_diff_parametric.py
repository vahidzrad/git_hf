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

from diffusion import Fracking





hsize =  1.5
mu_dynamic_list = [1.e-9]
pressure_steps_list= [1000]

for (k, pressure_steps) in enumerate(pressure_steps_list):
	for (j, mu_dynamic) in enumerate(mu_dynamic_list):
		       	Fracking(hsize, mu_dynamic, pressure_steps,j)








