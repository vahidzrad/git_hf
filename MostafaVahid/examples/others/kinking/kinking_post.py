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

from __future__ import division
from homepylab import *
from scipy.optimize import fsolve, minimize

# Numerical measured results
nums_G_1_5 = array([[0, 0], [0.2, 19.89], [0.4, 35.2], [0.6, 44.1], [0.8, 50.7], [1.0, 55.0]])
nums_G_1_1 = array([[0, 0], [0.2, 23.2], [0.4, 34.35], [0.6, 45.9], [0.8, 51.0], [1.0, 55.0]])
ell_0_0_5 = array([[0, 0], [0.2, 19.84], [0.4, 35.47], [0.6, 45.0], [0.8, 48.72], [1.0, 55.13]])

# Theoretic predictions

# Max hoop stress criterion
def max_hoop(K1, K2):
    if K2 == 0:
        return 0.0
    else:
        return 2*arctan(0.25*(K1/K2-sign(K2)*sqrt((K1/K2)**2+8)))/pi*180

# F matrix in [Amestoy and Leblond 1992]
F11 = lambda m: 1-3*pi**2/8*m**2+(pi**2-5*pi**4/128)*m**4+(pi**2/9-11*pi**4/72+\
                119*pi**6/15360)*m**6+5.07790*m**8-2.88312*m**10-0.0925*m**12+\
                2.996*m**14-4.059*m**16+1.63*m**18+4.1*m**20

F12 = lambda m: -3*pi/2*m+(10*pi/3+pi**3/16)*m**3+(-2*pi-133*pi**3/180+59*pi**5/1280)*m**5+\
                12.313906*m**7-7.32433*m**9+1.5793*m**11+4.0216*m**13-6.915*m**15+4.21*m**17+\
                4.56*m**19

F21 = lambda m: pi/2*m-(4*pi/3+pi**3/48)*m**3+(-2*pi/3+13*pi**3/30-59*pi**5/3840)*m**5-6.176023*m**7+\
                4.44112*m**9-1.5340*m**11-2.0700*m**13+4.684*m**15-3.95*m**17-1.32*m**19

F22 = lambda m: 1-(4+3*pi**2/8)*m**2+(8/3+29*pi**2/18-5*pi**4/128)*m**4+(-32/15-4*pi**2/9-1159*pi**4/7200+119*pi**6/15360)*m**6+\
                10.58254*m**8-4.78511*m**10-1.8804*m**12+7.280*m**14-7.591*m**16+0.25*m**18+12.5*m**20

# PLS
def PLS(K1, K2):
    f = lambda m: F21(m)/F22(m)+K2/K1
    alpha = fsolve(f, 0.0)[0]
    return alpha*180

# Gmax criterion
def Gmax(K1, K2):
    def f(m):
        newK1 = F11(m)*K1 + F12(m)*K2
        newK2 = F21(m)*K1 + F22(m)*K2
        return -newK1**2-newK2**2
    res = minimize(f, 0.8, bounds=[(0.0, 1.0)], method="L-BFGS-B")
    return res.x*180

# Generate the theoreic sequences
n = 100
K2 = linspace(0, 1, n)
theta_PLS = [0.0]
theta_hoop = [0.0]
theta_Gmax = [0.0]

for i in range(1, n):
    K1 = sqrt(2)/sqrt(1+K2[i]**2)
    theta_PLS.append(PLS(K1, -K2[i]*K1))
    theta_Gmax.append(Gmax(K1, -K2[i]*K1))
    theta_hoop.append(max_hoop(K1, -K2[i]*K1))

# Plots
plot(K2, theta_PLS, label="PLS")
plot(K2, theta_hoop, label=r"$\sigma_{\theta\theta}$-max")
plot(K2, theta_Gmax, label=r"$G$-max")
plot(nums_G_1_5[:, 0], nums_G_1_5[:, 1], 'o', label="$\ell=1\%$, $G_0=1.5$")
plot(ell_0_0_5[:, 0], ell_0_0_5[:, 1], 'o', label="$\ell=5\%$, $G_0=1.5$")
plot(nums_G_1_1[:, 0], nums_G_1_1[:, 1], 'v', label="$\ell=1\%$, $G_0=1.1$")
legend(loc="best")
xlabel("Prescribed $K_2/K_1$")
ylabel("Kinking angle / $^\circ$")
savefig("kinking.ipe")
