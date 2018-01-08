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

from edfpylab import *

def U(x, y, KI):
    E = 1
    nu = 0.2
    mu = E/(2*(1+nu))
    kappa = (3.0-nu)/(1.0+nu)
    r = sqrt(x**2+y**2)
    theta = arctan2(y, x)
    Ux = KI/(2*mu)*sqrt(r/(2*pi))*(kappa-cos(theta))*cos(theta/2)
    Uy = KI/(2*mu)*sqrt(r/(2*pi))*(kappa-cos(theta))*sin(theta/2)
    return [Ux, Uy]

def plotU(fname, color, text):
    data = loadtxt(fname, delimiter=",", skiprows=1)
    Ux, Uy, x, y = data[:, 0], data[:, 1], data[:, 5], data[:, 6]
    plot(y, Uy, color, label=text, markevery=int(len(y)/10))

# Slices for the displacement
# figure(1)
# plotU("data/uy_ell0x1.csv", "bo-", "$\ell=0.1$")
# plotU("data/uy_ell0x05.csv", "gv-", "$\ell=0.05$")
# plotU("data/uy_ell0x025.csv", "r>-", "$\ell=0.025$")
# y = linspace(-0.5, 0.5, 1000)
# plot(y, U(0, y, 1.0)[1], "k", linewidth=3.0, alpha=0.2, )
# xlabel("$y$-coordinate")
# ylabel("Displacement $u_y$")
# legend(loc="best")
# savefig("uy-ell.ipe")

# Slices for the damage
figure(2)
ell = 0.025
data = loadtxt("data/alpha.csv", delimiter=",", skiprows=1)
alpha, y = data[:, 0], data[:, 4]
imax = argmax(alpha)
y = y-y[imax]
plot(y[imax-48:imax+48]/ell, alpha[imax-48:imax+48], label="Num.")
D = 2.0
y = linspace(-D, D, 101)
plot(y, (1-abs(y)/D)**2, label="Theo.")
xlabel("$y/\ell$")
ylabel(r"Damage $\alpha$")
legend(loc="best")
savefig("alpha.ipe", format="ipe")