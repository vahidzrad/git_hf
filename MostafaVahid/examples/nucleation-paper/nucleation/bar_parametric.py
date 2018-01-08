# Copyright (C) 2015 Corrado Maurini, Tianyi Li
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
from bar_basic import *

law_list = ["AT1", "AT2"]
k_list = [1.0, 2.0, 5.0, 10.0]
ell_list = pl.logspace(-2, -1, 5)

# Using constitutive laws AT1 and AT2
for law in law_list:

    # Using different internal lengths
    for ell in ell_list:

        problem = NucleationBar(ell, law=law)
        problem.print0("\033[1;35m--------------------------------------\033[1;m")
        problem.print0("\033[1;35mSolving %s model with ell = %.1e -\033[1;m" %(law, ell))
        problem.print0("\033[1;35m--------------------------------------\033[1;m")
        problem.solve()

# Using constitutive laws ATk
for k in k_list:

    # Using different internal lengths
    for ell in ell_list:

        problem = NucleationBar(ell, law="ATk", k=k)
        problem.print0("\033[1;35m----------------------------------------\033[1;m")
        problem.print0("\033[1;35mSolving ATk model with k = %g and ell = %.1e -\033[1;m" %(k, ell))
        problem.print0("\033[1;35m----------------------------------------\033[1;m")
        problem.solve()

# Remove the .pyc file
MPI.barrier(mpi_comm_world())
if MPI.rank(mpi_comm_world()) == 0:
    os.remove("bar_basic.pyc")