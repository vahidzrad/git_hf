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
import pylab as pl
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


def find_data(items, values):

    for subdir, dirs, files in os.walk("hole-results-old/"):

        if not os.path.isfile(subdir + "/parameters.xml"):
            continue

        # Extract the parameters used
        param = Parameters()
        File(subdir + "/parameters.xml") >> param

        # It matched, extract information
        if pl.allclose(pl.array([eval("param." + item) for item in items]), pl.array(values)):
            data = pl.loadtxt(subdir + "/energies.txt")
            t = data[:, 0]
            S = data[:, 2]

            # Umtimate stress
            tf = (t[-1] + t[-2])/2

            # Elastic limit
            ind = pl.find(S > 0)[0]
            te = (t[ind] + t[ind-1])/2

            return te, tf

def ConvergenceMesh():
    pl.figure()
    Roell_list = pl.logspace(pl.log10(0.1), pl.log10(50), 20)

    tf_list = []
    te_list = []
    for Roell in Roell_list:
        ell = 1/Roell
        te, tf = find_data(["material.ell", "problem.rho", "problem.hsize"], [ell, 0.1, 1e-3])
        te_list.append(te); tf_list.append(tf)
    pl.loglog(Roell_list, tf_list, "bo-", alpha=0.5, label=r"$h=1\times10^{-3}$")

    tf_list = []
    te_list = []
    for Roell in Roell_list:
        ell = 1/Roell
        te, tf = find_data(["material.ell", "problem.rho", "problem.hsize"], [ell, 0.1, 4e-3])
        te_list.append(te); tf_list.append(tf)
    pl.loglog(Roell_list, tf_list, "g*-", alpha=0.5, label=r"$h=2\times10^{-3}$")


    tf_list = []
    te_list = []
    for Roell in Roell_list:
        ell = 1/Roell
        te, tf = find_data(["material.ell", "problem.rho", "problem.hsize"], [ell, 0.1, 1e-2])
        te_list.append(te); tf_list.append(tf)
    pl.loglog(Roell_list, tf_list, "r>-", alpha=0.5, label=r"$h=1\times10^{-2}$")

    pl.xlabel(r"Relative defect size $a/\ell$")
    pl.ylabel("Remote stress at fracture $\sigma_\infty/\sigma_0$")
    pl.legend(loc="best")
    pl.title(r"$\rho=0.1$").set_y(1.04)
    pl.xlim([9e-2, 1e2])
    pl.grid(True)
    pl.savefig("conv_h.pdf",bbox_inches='tight')

def Circular():
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([9e-2, 1e2])

    # Elastic limit
    Roell_list = pl.logspace(pl.log10(0.1), pl.log10(50), 20)
    ax.plot([Roell_list[0], Roell_list[-1]], [1/3, 1/3], "k", linewidth=3.0, alpha=0.2)

    # Numerical results
    tf_list = []
    te_list = []
    for Roell in Roell_list:
        ell = 1/Roell
        te, tf = find_data(["material.ell", "problem.rho", "problem.hsize"], [ell, 1, 1e-3])
        te_list.append(te); tf_list.append(tf)
    ax.plot(Roell_list, tf_list, "bo-")  # label="Num."

    ind = int(len(Roell_list)/2.5)
    coeff = pl.polyfit(pl.log(Roell_list[ind:-ind]), pl.log(tf_list[ind:-ind]), 1)
    poly = pl.poly1d(coeff)
    fit = pl.exp(poly(pl.log(Roell_list[ind:-ind])))
    print("The slope = %.2e" %(coeff[0]))
    pl.loglog(Roell_list[ind:-ind], fit, "k-", linewidth=3.0, alpha=0.2)

    # Experimental results
    E = 3e3
    sigc = 72
    Gc = 290e-3
    ell = 3/8*Gc*E/sigc**2
    print ell
    ell = 26e-3
    data = pl.loadtxt("literature/exp.csv", delimiter=",")
    Roell_exp = data[:, 0]/(2*ell); Roell_exp[0] = Roell_list[0]
    sig_center = (pl.amin(data[:, 1:], 1)+pl.amax(data[:, 1:], 1))/2
    err1 = -(pl.amin(data[:, 1:], 1)-sig_center)/sigc
    err2 = (pl.amax(data[:, 1:], 1)-sig_center)/sigc
    pl.errorbar(Roell_exp, sig_center/sigc, yerr=[err1, err2], label="Exp.", fmt="g.")

    # Numerical results from C. Kuhn
    # data = loadtxt("literature/Kuhn.csv", delimiter=",")
    # # ell = 0.00885
    # Roell_Kuhn = data[:, 0]/ell
    # ax.plot(Roell_Kuhn, data[:, 1], "r>-", label="Kuhn")

    pl.ylim([1.0/4.0, 1.1])
    pl.xlabel("Relative hole size $R/\ell$")
    pl.ylabel("$\sigma/\sigma_0$ at fracture")
    pl.legend(loc="best")
    pl.savefig("circular_tf.pdf",bbox_inches='tight')

def Elliptic():

    rho_list = pl.logspace(-1, 0, 2)
    Roell_list = pl.logspace(pl.log10(0.1), pl.log10(50), 20)
    markers = ["bo-", "gv-", "r>-"]

    # Elastic limit
    pl.plot([Roell_list[0], Roell_list[-1]], [1/3, 1/3], "k", linewidth=3.0, alpha=0.2)

    for i, rho in enumerate(rho_list):
        tf_list = []
        te_list = []
        for Roell in Roell_list:
            ell = 1/Roell
            te, tf = find_data(["material.ell", "problem.rho", "problem.hsize"], [ell, rho, 1e-3])
            te_list.append(te); tf_list.append(tf)
        pl.savetxt("Rolist%s.txt"%i,Roell_list)
        pl.savetxt("tf_list%s.txt"%i,tf_list)
        pl.savetxt("te_list%s.txt"%i,te_list)

        pl.figure(1)
        pl.loglog(Roell_list, tf_list, markers[i], label=r"$\rho=%g$" %(rho))
        pl.savefig("elliptic_tf_%s.pdf"%i)
        pl.figure(2)
        pl.loglog(Roell_list, te_list, markers[i], label=r"$\rho=%g$" %(rho))
        pl.savefig("elliptic_te_%s.pdf"%i)

        if near(rho, 0.1):
            pl.figure(1)
            ind = 12
            coeff = pl.polyfit(pl.log(Roell_list[ind:]), pl.log(tf_list[ind:]), 1)
            poly = pl.poly1d(coeff)
            fit = pl.exp(poly(pl.log(Roell_list[ind:])))
            print("The slope = %.2e" %(coeff[0]))
            pl.loglog(Roell_list[ind:], fit, "k-", linewidth=3.0, alpha=0.2)

    pl.figure(1)
    pl.xlabel(r"Relative defect size $a/\ell$")
    pl.ylabel("$\sigma/\sigma_0$ at fracture")
    pl.legend(loc="best")
    pl.xlim([9e-2, 1e2])
    pl.grid(True)
    pl.savefig("elliptic_tf.pdf")

    pl.figure(2)
    pl.xlabel(r"Relative defect size $a/\ell$")
    pl.ylabel("$\sigma/\sigma_0$ at loss of elasticity")
    pl.legend(loc="best")
    pl.xlim([9e-2, 1e2])
    pl.grid(True)
    pl.savefig("elliptic_te.pdf")

    pl.figure(3)
    rho_list = pl.linspace(0.1, 1, 10)
    te_list = []
    for i, rho in enumerate(rho_list):
        te, tf = find_data(["material.ell", "problem.rho", "problem.hsize"], [1.0, rho, 1e-3])
        te_list.append(te)
    pl.plot(rho_list, te_list, "o", label="Num.")
    pl.savetxt("rho_list.txt",rho_list)
    pl.savetxt("rho_te_list.txt",te_list)
    rho_list = pl.linspace(0.1, 1, 1000)
    pl.plot(rho_list, rho_list/(rho_list+2), "k", label="Theo.", linewidth=3.0, alpha=0.2)
    pl.xlabel(r"Ellipticity $\rho$")
    pl.ylabel("$\sigma/\sigma_0$ at loss of elasticity")
    pl.legend(loc="best")
    pl.grid(True)
    pl.savefig("elliptic_te_rho.pdf",bbox_inches='tight')

def EllipticSlope():

    rho_list = [0.5]
    Roell_list = pl.logspace(pl.log10(0.1), pl.log10(50), 20)
    markers = ["bo-", "gv-", "r>-"]

    for i, rho in enumerate(rho_list):
        tf_list = []
        te_list = []
        for Roell in Roell_list:
            ell = 1/Roell
            te, tf = find_data(["material.ell", "problem.rho", "problem.hsize"], [ell, rho, 1e-3])
            te_list.append(te); tf_list.append(tf)

        pl.figure(i)
        pl.loglog(Roell_list, tf_list, markers[i], label=r"$\rho=%g$" %(rho))

        ind1 = 8
        ind2 = 5
        coeff = pl.polyfit(pl.log(Roell_list[ind1:-ind2]), pl.log(tf_list[ind1:-ind2]), 1)
        poly = pl.poly1d(coeff)
        fit = pl.exp(poly(pl.log(Roell_list[ind1:-ind2])))
        print("The slope = %.2e" %(coeff[0]))
        pl.loglog(Roell_list[ind1:-ind2], fit, "k-", linewidth=3.0, alpha=0.2)

        pl.xlabel(r"Relative defect size $a/\ell$")
        pl.ylabel("$\sigma/\sigma_0$ at fracture")
        pl.legend(loc="best")
        pl.xlim([9e-2, 1e2])
        pl.grid(True)
        pl.savefig("elliptic_rho_" + str(rho).replace(".", "x") + ".pdf",bbox_inches='tight')

#ConvergenceMesh()
#Circular()
Elliptic()
#EllipticSlope()
