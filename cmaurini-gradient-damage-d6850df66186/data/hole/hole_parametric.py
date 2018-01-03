# Copyright (C) 2017 Tianyi Li, Corrado Maurini
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
from hole_basic import *

if __name__ == '__main__':

    Roell_list = pl.logspace(pl.log10(0.1), pl.log10(50), 20)
    rho_list = pl.linspace(0.1, 1.0, 10)[1:-1]
    for rho in rho_list:
        for Roell in Roell_list:
            ell = 1.0/Roell
            problem = EllipticHole(ell=ell, rho=rho)
            problem.print0("\033[1;35m--------------------------------------\033[1;m")
            problem.print0("\033[1;35m- Solving with rho = %.1e and ell = %.1e-\033[1;m" %(rho, ell))
            problem.print0("\033[1;35m--------------------------------------\033[1;m")
            problem.solve()

    # rho_list = pl.linspace(0.1, 1.0, 10)
    # for rho in rho_list:
    #     ell = 1.0
    #     problem = EllipticHole(ell=ell, rho=rho)
    #     problem.print0("\033[1;35m--------------------------------------\033[1;m")
    #     problem.print0("\033[1;35m- Solving with rho = %.1e and ell = %.1e-\033[1;m" %(rho, ell))
    #     problem.print0("\033[1;35m--------------------------------------\033[1;m")
    #     problem.solve()

    # Remove the .pyc file
    MPI.barrier(mpi_comm_world())
    if MPI.rank(mpi_comm_world()) == 0:
        os.remove("hole_basic.pyc")
