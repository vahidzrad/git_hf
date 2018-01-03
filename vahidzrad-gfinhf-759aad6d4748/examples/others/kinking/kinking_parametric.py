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

from dynamic_gradient_damage import *
from kinking_basic import *
import os

# Varying the internal length
for ell in [0.05]:

    # Varying the K2/K1 factor
    for K2 in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        problem = Kinking(K2=K2, ell=ell)
        problem.solve()

#### Remove the .pyc file ####
MPI.barrier(mpi_comm_world())
if MPI.rank(mpi_comm_world()) == 0:
    os.remove("kinking_basic.pyc")