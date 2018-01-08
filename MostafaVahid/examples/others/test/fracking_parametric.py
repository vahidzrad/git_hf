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

from gradient_damage import *
from fracking_basic import *
import os

hsize_list = [1.0e-2, 6.7e-3, 5.7e-3, 5.0e-3]
ell_list = [6.7e-3, 1.0e-2, 1.3e-2, 2.0e-2, 2.7e-2, 3.0e-2]

# Using different internal lengths
for ell in ell_list:

    # Varying the hsize mesh size
    for hsize in hsize_list:
        problem = Fracking(hsize=hsize, ell=ell)
        problem.solve()

#### Remove the .pyc file ####
MPI.barrier(mpi_comm_world())
if MPI.rank(mpi_comm_world()) == 0:
    os.remove("fracking_basic.pyc")
