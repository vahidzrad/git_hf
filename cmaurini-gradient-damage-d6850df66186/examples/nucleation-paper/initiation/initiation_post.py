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

from __future__ import division
from fenics import *
from homepylab import *
import os

def find_data(items, values):

    for subdir, dirs, files in os.walk("initiation-results/"):

        if not os.path.isfile(subdir + "/parameters.xml"):
            continue

        # Extract the parameters used
        param = Parameters()
        File(subdir + "/parameters.xml") >> param

        # It matched, extract information
        items_num = []
        values_num = []
        items_str = []
        values_str = []
        for i, value in enumerate(values):
            if type(value) == float or type(value) == int:
                items_num.append(items[i])
                values_num.append(value)
            else:
                items_str.append(items[i])
                values_str.append(value)

        val_items_str = [eval("param." + item) for item in items_str]
        val_items_num = [eval("param." + item) for item in items_num]
        if val_items_str == values_str and allclose(array(val_items_num), array(values_num)):
            data = loadtxt(subdir + "/Gtheta.txt")
            t = data[:, 0]
            G = data[:, 1]

            if os.path.isfile(subdir + "/Rayleigh.txt"):
                data = loadtxt(subdir + "/rayleigh.txt")
                rq = data[:, 1]
                return t, G, rq
            else:
                return t, G

# for u_degree in [1, 2]:
#     t, G = find_data(["problem.desc", "fem.u_degree", "problem.hsize"], ["crack", u_degree, 0.01])
#     Gcnum = 1+3.0/8.0/5.0
#     plot(t, G/Gcnum, "o-", markevery=int(len(t)/10.0), label="P%d, $\ell/h=5$" %(u_degree))

# ylim([0, 2])
# xlabel("Imposed $G$")
# ylabel(r"$G/(G_\mathrm{c})_\mathrm{eff}$")
# legend(loc="best")
# savefig("initiation_p.ipe")

# figure()
# for hsize in [0.01, 0.005, 0.0025]:
#     t, G = find_data(["problem.desc", "fem.u_degree", "problem.hsize", "problem.stability"], ["crack", 1, hsize, False])
#     Gcnum = 1+3.0/8.0*(hsize/0.05)
#     plot(t, G/Gcnum, "v-", markevery=int(len(t)/10.0), label="P1, $\ell/h=%d" %(0.05/hsize))

# ylim([0, 2])
# xlabel("Imposed $G$")
# ylabel(r"$G/(G_\mathrm{c})_\mathrm{eff}$")
# legend(loc="best")
# savefig("initiation_h.ipe")

fig = figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
Gcnum = 1+3.0/8.0*(0.005/0.05)
t, G, rq = find_data(["problem.desc", "fem.u_degree", "problem.hsize", "problem.stability"], ["crack", 1, 0.005, True])
ax1.plot(t, G/Gcnum, "bo-", markevery=int(len(t)/10.0), label="$G$")

ind = find(rq < 2)[0]
ax2.plot(t[ind:], rq[ind:], "gv-", markevery=int(len(t)/10.0), label="R")

ylim([0, 2])
ax1.set_xlabel("Imposed $G$")
ax1.set_ylabel(r"$G/(G_\mathrm{c})_\mathrm{eff}$")
ax2.set_ylabel("Rayleigh")
title("P1, $\ell/h=10").set_y(1.04)
savefig("initiation_r.ipe")