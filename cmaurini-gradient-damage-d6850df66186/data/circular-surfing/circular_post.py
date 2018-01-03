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

import matplotlib
matplotlib.use("module://backend_ipe")
from gradient_damage import *
matplotlib.rcParams["ipe.stylesheet"] = "/home/TL0E122N/.local/share/ipe/7.1.7/styles/basic.isy"
matplotlib.rcParams["ipe.preamble"] = "\usepackage{newtxtext,newtxmath}"
matplotlib.rcParams["ipe.textsize"] = True
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

    for subdir, dirs, files in os.walk("circular-results/"):

        if not os.path.isfile(subdir + "/parameters.xml"):
            continue

        # Extract the parameters used
        File(subdir + "/parameters.xml") >> parameters

        # It matched, extract information
        if [eval("parameters." + item) for item in items] == values:
            data = pl.loadtxt(subdir + "/energies.txt")
            t, E, S = data[:, 0], data[:, 1], data[:, 2]
            Gtheta = pl.loadtxt(subdir + "/Gtheta.txt")

            return t, E, S, Gtheta, parameters

# nu = 0
def nu_0():

    t, E, S, Gtheta, p = find_data(["material.nu"], [0.0])
    Gc_num = cal_Gc_num(p.material.law, p.problem.hsize, p.material.ell, p.material.Gc)
    l = S/Gc_num
    R = 1
    theta_num = l/R

    fig = pl.figure(1)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(t, theta_num)
    coeff = pl.polyfit(t, theta_num, 1)
    poly = pl.poly1d(coeff)
    fit = poly(t)
    ax1.plot(t, fit, linewidth=3, alpha=0.2)
    print("Slope is %.3e" % (coeff[0]))

    ax1.set_xlabel("Prescribed crack angle (rad)")
    ax1.set_ylabel("Computed crack angle (rad)")
    
    ax2.plot(t, Gtheta, "*-", markevery=10)
    ax2.set_ylabel(r"Energy release rate $G^\alpha/(G_\mathrm{c})_\mathrm{eff}$")
    
    pl.savefig("nu_0.ipe")

nu_0()
