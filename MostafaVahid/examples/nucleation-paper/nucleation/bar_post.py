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

import matplotlib
matplotlib.use("module://backend_ipe")
from gradient_damage import *
matplotlib.rcParams["ipe.stylesheet"] = "/home/TL0E122N/.local/share/ipe/7.1.7/styles/basic.isy"
matplotlib.rcParams["ipe.preamble"] = "\usepackage{newtxtext,newtxmath}"
matplotlib.rcParams["ipe.textsize"] = True
import os

def find_data(items, values):

    for subdir, dirs, files in os.walk("bar-results/"):

        if not os.path.isfile(subdir + "/parameters.xml"):
            continue

        # Extract the parameters used
        File(subdir + "/parameters.xml") >> parameters
        law = parameters.material.law
        k = parameters.material.k
        ell = parameters.material.ell

        flag = 0
        for (item, value) in zip(items, values):
            if isinstance(value, float):
                if abs(eval(item)-value)/abs(value) > 1e-3:
                    flag = flag - 1
            else:
                if eval(item) != value:
                    flag = flag - 1

        if flag == 0:
            data = pl.loadtxt(subdir + "/F.txt")
            U = data[:, 0]  # /pl.sqrt(ell)
            F = data[:, 1]  # /pl.sqrt(ell)
            return U, F

# Plot AT1 and AT2 results
law_list = ["AT1", "AT2"]
ell_list = pl.logspace(-2, -1, 5)
color_list = ["b", "g", "r", "c"]

for (i, law) in enumerate(law_list):
    color = color_list[i]
    Um_list = pl.zeros(5)

    # Reference results
    # U_ref, F_ref = find_data(["ell", "law"], [0.01, law])

    for (j, ell) in enumerate(ell_list):
        U, F = find_data(["ell", "law"], [ell, law])

        # Renormalize the results
        ind = pl.argmax(F)
        U = U/U[ind]
        F = F/max(F)

        Um_list[j] = U[-1]

        pl.figure(i+2)
        pl.plot(U, F, label="$\ell/L=%.2f$" %(ell))

    coeff = pl.polyfit(pl.log(ell_list), pl.log(Um_list), 1)
    poly = pl.poly1d(coeff)
    fit = pl.exp(poly(pl.log(ell_list)))
    print("The slope for law %s is %.2e" %(law, coeff[0]))
    pl.figure(1)
    pl.loglog(ell_list, Um_list, color+"o", label="%s" % (law))
    pl.loglog(ell_list, fit, color+"-")

pl.figure(1)
pl.xlabel("Internal length $\ell/L$")
pl.ylabel("Ultimate disp. $U_\mathrm{m}/(U_\mathrm{m})_\mathrm{ref}$")
pl.title("AT1 and AT2 model").set_y(1.04)
pl.legend(loc="best")
pl.grid(True)
pl.savefig("bar-Um-ell-AT12.ipe", format="ipe")

for (i, law) in enumerate(law_list):
    pl.figure(i+2)
    pl.xlabel("Displacement $U/U_\mathrm{ref}$")
    pl.ylabel("Stress $\sigma/\sigma_\mathrm{ref}$")
    pl.title("%s model" % (law)).set_y(1.04)
    pl.legend(loc="best")
    pl.savefig("bar-%s.ipe" % (law), format="ipe")

# Plot ATk results
k_list = [1.0, 2.0, 5.0, 10.0]

for (i, k) in enumerate(k_list):
    color = color_list[i]
    Um_list = pl.zeros(5)

    # Reference results
    # U_ref, F_ref = find_data(["ell", "law", "k"], [0.01, "ATk", k])

    for (j, ell) in enumerate(ell_list):
        U, F = find_data(["ell", "law", "k"], [ell, "ATk", k])

        # Renormalize the results
        # U = U/(max(U_ref)-1e-4)
        # F = F/max(F_ref)

        Um_list[j] = U[-1]

        pl.figure(i+5)
        pl.plot(U, F, label="$\ell/L=%.2f$" % (ell))

    coeff = pl.polyfit(pl.log(ell_list), pl.log(Um_list), 1)
    poly = pl.poly1d(coeff)
    fit = pl.exp(poly(pl.log(ell_list)))
    print("The slope for k = %d is %.2e" %(k, coeff[0]))
    pl.figure(4)
    pl.loglog(ell_list, Um_list, color+"o", label="$k=%d$" % (k))
    pl.loglog(ell_list, fit, color+"-")

pl.figure(4)
pl.xlabel("Internal length $\ell/L$")
pl.ylabel("Ultimate disp. $U_\mathrm{m}/U_\mathrm{ref}(\ell,L)$")
pl.title("ATk model").set_y(1.04)
pl.legend(loc="best")
pl.grid(True)
pl.savefig("bar-Um-ell-ATk.ipe", format="ipe")

for (i, k) in enumerate(k_list):
    pl.figure(i+5)
    pl.xlabel("Displacement $U/U_\mathrm{ref}$")
    pl.ylabel("Stress $\sigma/\sigma_\mathrm{ref}$")
    pl.title("$k=%d$" % (k)).set_y(1.04)
    pl.legend(loc="best")
    pl.savefig("bar-k%d.ipe" % (k), format="ipe")