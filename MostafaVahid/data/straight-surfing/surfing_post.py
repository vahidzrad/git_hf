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

from homepylab import *
from fenics import *
import pylab as pl
import os

def cal_Gc_num(law, hsize, ell, Gc):
    if law == "AT1":
        Gc_num = (1+3*hsize/(8*ell)) * Gc
    elif law == "AT2":
        Gc_num = (1+hsize/(2*ell)) * Gc
    elif law == "ATk":
        Gc_num = (1+hsize/(pl.pi*ell)) * Gc

    return Gc_num

def find_data(items, values):

    for subdir, dirs, files in os.walk("surfing-results/"):

        if not os.path.isfile(subdir + "/parameters.xml"):
            continue

        # Extract the parameters used
        File(subdir + "/parameters.xml") >> parameters
        hsize = parameters.problem.hsize
        law = parameters.material.law
        k = int(parameters.material.k)
        ell = parameters.material.ell
        Gc = parameters.material.Gc

        flag = 0
        for (item, value) in zip(items, values):
            if eval(item) != value:
                flag = flag - 1

        if flag == 0:
            data = pl.loadtxt(subdir + "/energies.txt")
            t, E, S = data[:, 0], data[:, 1], data[:, 2]
            Gc_num = cal_Gc_num(law, hsize, ell, Gc)
            Gtheta = pl.loadtxt(subdir + "/Gtheta.txt")

            return Gc_num, t, E, S, Gtheta

# AT1 for a fixed ratio of ell/h=5 while varying ell
fig1 = pl.figure(1)
ax1 = fig1.add_subplot(111)
ax2 = ax1.twinx()
markers = ["o", "v", ">"]
for i, ell in enumerate([0.025, 0.05, 0.1]):
    Gc_num, t, E, S, Gtheta = find_data(["law", "ell/hsize", "ell"], ["AT1", 5, ell])
    ax1.plot(t, S/Gc_num, markers[i]+"-", label="$\ell=%g$" %(ell), markevery=int(len(t)/10))
    ax2.plot(t, Gtheta/Gc_num, markers[i]+"-", label="$\ell=%g$" %(ell), markevery=int(len(t)/10))

ax1.set_xlabel("Time")
ax1.set_ylabel("Crack length $\mathcal{S}/(G_\mathrm{c})_\mathrm{eff}$")
ax2.set_ylabel(r"$G/(G_\mathrm{c})_\mathrm{eff}$")
pl.legend(loc="lower right")
pl.savefig("AT1_fixed_ratio.ipe", format="ipe")

# AT1 for a fixed ell while varying h
# fig2 = pl.figure(2)
# ax1 = fig2.add_subplot(111)
# ax2 = ax1.twinx()
# for i, hsize in enumerate([0.01, 0.02, 0.04]):
#     Gc_num, t, E, S, Gtheta = find_data(["law", "ell", "hsize"], ["AT1", 0.05, hsize])
#     ax1.plot(t, S/Gc_num, markers[i]+"-", label="$h=%g$" %(hsize), markevery=int(len(t)/10))
#     ax2.plot(t, Gtheta/Gc_num, markers[i]+"-", label="$h=%g$" %(hsize), markevery=int(len(t)/10))
#
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Crack length $\mathcal{S}/(G_\mathrm{c})_\mathrm{eff}$")
# ax2.set_ylabel(r"$G/(G_\mathrm{c})_\mathrm{eff}$")
# pl.legend(loc="lower right")
# pl.savefig("AT1_ell_0_05.ipe", format="ipe")

# ATk for a fixed ell and h while varying k
# fig3 = pl.figure(3)
# ax1 = fig3.add_subplot(111)
# ax2 = ax1.twinx()
# for k in [1, 2, 4, 10]:
#     Gc_num, t, E, S, Gtheta = find_data(["law", "ell", "hsize", "k"], ["ATk", 0.025, 0.005, k])
#     ax1.plot(t, S/Gc_num, label="$k=%g$" %(k))
#     ax2.plot(t, Gtheta/Gc_num, label="$k=%g$" %(k))

# ax1.set_xlabel("Time")
# ax1.set_ylabel("Crack length $\mathcal{S}/(G_\mathrm{c})_\mathrm{eff}$")
# ax2.set_ylabel(r"$G/(G_\mathrm{c})_\mathrm{eff}$")
# pl.legend(loc="lower right")
# pl.title(r"ATk model using a fixed $\ell=0.025$ and $h=5\times 10^{-3}$").set_y(1.04)
# pl.savefig("ATk_fixed_ell_and_h.ipe", format="ipe")
