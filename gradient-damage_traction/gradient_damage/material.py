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
from fenics import *
import sympy

class GradientDamageMaterial(object):

    """
    This class specifies the material model.
    It includes the definition of the strain, the stress, and the internal energy density

    The Material includes a subdomain_id (size_t) that is used to identify the subdomain where
    the material will be applied, in the case of a multimaterial model.
    """

    def __init__(self, material_parameters, subdomain_id="everywhere"):
        """
        Optional input:
            - material parameters (Parameter)
            - subdomain_id (size_t)
        """
        mp = material_parameters
        self.law = mp.law
        self.k = Constant(mp.k)  # k parameter for the "ATk" model
        self.kres = Constant(mp.kres)
        self.E = Constant(mp.E)
        self.nu = Constant(mp.nu)
        self.Gc = Constant(mp.Gc)
        self.ell = Constant(mp.ell)
        self.mu = self.E/(2.0*(1.0+self.nu))
        if not mp.pstress:  # plane strain
            self.lmbda = self.E*self.nu/((1.0+self.nu)*(1.0-2.0*self.nu))
        else:  # plane stress
            self.lmbda = self.E*self.nu/(1.0-self.nu**2)
        self._eps0 = self.eps0()
        self.subdomain_id = subdomain_id

    def a(self, alpha):
        """
        Modulation of the elastic stiffness
        """
        if self.law == "AT1":
            return (1-alpha)**2
        elif self.law == "AT2":
            return (1-alpha)**2
        elif self.law == "ATk":
            return (1-self.w(alpha))/(1+(self.k-1)*self.w(alpha))

    def w(self, alpha):
        """
        Local energy dissipation
        """
        if self.law == "AT1":
            return alpha
        elif self.law == "AT2":
            return alpha**2
        elif self.law == "ATk":
            return 1-(1-alpha)**2

    def eps(self, u):
        """
        Geometrical strain
        """
        return sym(grad(u))

    def eps0(self):
        """
        Inelastic strain, supposed to be isotropic
        """
        return 0.

    def epse(self, u):
        """
        Elastic strain
        """
        Id = Identity(len(u))
        return self.eps(u) - self._eps0*Id

    def sigma0(self, u):
        """
        Application of the sound elasticy tensor on the strain tensor
        """
        Id = Identity(len(u))
        return 2.0*self.mu*self.epse(u) + self.lmbda*tr(self.epse(u))*Id


    def sigma(self, u, alpha):
        """
        Stress
        """
        return (self.a(alpha)+self.kres) * self.sigma0(u)

    def elastic_energy_density(self, u, alpha):
        """
        Elastic energy density
        """
        return 0.5 * inner(self.sigma(u, alpha), self.epse(u))

    def dissipated_energy_density(self, u, alpha):
        """
        Dissipated energy density
        """
        z = sympy.Symbol("z", positive=True)
        self.c_w = float(4*sympy.integrate(sympy.sqrt(self.w(z)), (z, 0, 1)))
        return self.Gc/self.c_w * (self.w(alpha)/self.ell + self.ell*inner(grad(alpha), grad(alpha)))

    def user_energy(self, u, alpha):
        """
        Specify here a user energy density to add to the other terms, for example bulk forces
        """
        return 0.

    def rqP_density(self, u, alpha, v, beta):
        """
        Rayleigh ratio = rqP / rqN, where P'' = rqP - rqN
        """
        return inner(sqrt(self.a(alpha))*self.sigma0(v) + diff(self.a(alpha), alpha)/sqrt(self.a(alpha))*beta*self.sigma0(u), sqrt(self.a(alpha))*self.epse(v) + diff(self.a(alpha), alpha)/sqrt(self.a(alpha))*beta*self.epse(u)) + 2*self.Gc/self.c_w*self.ell*inner(grad(beta), grad(beta))

    def rqN_density(self, u, alpha, v, beta):
        """
        Rayleigh ratio = rqP / rqN, where P'' = rqP - rqN
        """
        ahat = 0.5*(-diff(diff(self.a(alpha), alpha), alpha) + 2*diff(self.a(alpha), alpha)**2/self.a(alpha))
        return (ahat*inner(self.sigma0(u), self.epse(u)) - self.Gc/(self.c_w*self.ell)*diff(diff(self.w(alpha), alpha), alpha))*beta**2
