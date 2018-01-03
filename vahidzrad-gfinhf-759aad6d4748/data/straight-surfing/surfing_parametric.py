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
from surfing_basic import *

# Fix an h/ell ratio (1/5) and vary the internal lengths
def fixed_h_ell_vary_ell(hsize_list=[0.005, 0.01, 0.02], ell_list=[0.025, 0.05, 0.1]):
    for (hsize, ell) in zip(hsize_list, ell_list):
        problem = Surfing(hsize, ell)
        problem.print0("\033[1;35m----------------------------------------------------\033[1;m")
        problem.print0("\033[1;35m- Solving surfing problem with h = %g and ell = %g -\033[1;m" %(hsize, ell))
        problem.print0("\033[1;35m----------------------------------------------------\033[1;m")
        problem.solve()

# Fix an internal length (0.05) and vary the mesh size h
def fixed_ell_vary_h():
    hsize_list = [0.04]  # , 0.02]  # 0.01 already done
    ell_list = [0.05]  # , 0.05]
    for (hsize, ell) in zip(hsize_list, ell_list):
        problem = Surfing(hsize, ell)
        problem.print0("\033[1;35m----------------------------------------------------\033[1;m")
        problem.print0("\033[1;35m- Solving surfing problem with h = %g and ell = %g -\033[1;m" %(hsize, ell))
        problem.print0("\033[1;35m----------------------------------------------------\033[1;m")
        problem.solve()

# Study the parameter k
def study_of_k():
    hsize = 0.005
    ell = 0.025
    for k in [4.0, 10.0]:  # [1.0, 2.0, 4.0, 10.0]:
        problem = Surfing(hsize, ell, "ATk", k)
        problem.print0("\033[1;35m----------------------------------------------------\033[1;m")
        problem.print0("\033[1;35mSolving surfing problem with h = %g and ell = %g\033[1;m" %(hsize, ell))
        problem.print0("\033[1;35m----------------------------------------------------\033[1;m")
        problem.solve()

study_of_k()

# Remove the .pyc file
MPI.barrier(mpi_comm_world())
if MPI.rank(mpi_comm_world()) == 0:
    os.remove("surfing_basic.pyc")