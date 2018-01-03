# Copyright (C) 2015 Tianyi Li
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

from pylab import *
from gradient_damage import *
import os

def find_data(items, values):

    for subdir, dirs, files in os.walk("stability1d-results/"):

        if not os.path.isfile(subdir + "/parameters.xml"):
            continue

        # Extract the parameters used
        File(subdir + "/parameters.xml") >> parameters

        # It matched, extract information
        if [eval("parameters." + item) for item in items] == values:
            data = pl.loadtxt(subdir + "/rayleigh.txt")
            return data

def AT1_stability_1d():

    ell_list = pl.linspace(0.1, 1.0, 10)
    U = []
    for ell in ell_list:
        data = find_data(["material.ell", "material.law"], [ell, "AT1"])
        U.append(data[-1, 0])

    pl.plot(U, 1/ell_list, "-o", label="Num.")

    lmbda = sqrt(2.0/3.0)*pl.pi
    pl.plot([1, 1], [lmbda, 10], "g-", label="Theo.", linewidth=2, alpha=0.5)
    Ut = pl.linspace(1, 3, 100)
    pl.plot(Ut, lmbda/Ut, "g-", linewidth=2, alpha=0.5)
    pl.xlabel("Unstable displacement $U/U_\mathrm{e}$")
    pl.ylabel("Relative length $L/\ell$")
    pl.legend(loc="best")
    pl.xlim([0, 3])
    pl.ylim([0, max(1/ell_list)])
    pl.savefig("1d_stability.pdf")

def ATk_stability_1d():

    k_list = pl.linspace(2, 12, 6)
    ell_list = pl.linspace(0.1, 1.0, 10)
    for k in k_list:
        U = []
        for ell in ell_list:
            data = find_data(["material.ell", "material.law", "material.k"], [ell, "ATk", k])
            U.append(data[-1, 0])

        pl.plot(U, 1/ell_list, "-o", label="$k=%g$" % (k))

    pl.xlabel("Unstable displacement $U/U_\mathrm{e}$")
    pl.ylabel("Relative length $L/\ell$")
    pl.legend(loc="best")
    pl.xlim([0, 3])
    pl.ylim([0, max(1/ell_list)])
    pl.savefig("atk_stability.pdf")

AT1_stability_1d()
