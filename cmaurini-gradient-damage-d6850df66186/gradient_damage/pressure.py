# Copyright (C) 2016 Tianyi Li
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

from fenics import *

class Gtheta(object):

    """
    G-theta method for gradient damage models
    """

    def calc_theta(self):

        x0 = self.x0
        y0 = self.y0
        r = self.r
        R = self.R
        def neartip(x, on_boundary):
            dist = sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
            return dist < r

        def outside(x, on_boundary):
            dist = sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
            return dist > R

        class bigcircle(SubDomain):
            def inside(self, x, on_boundary):
                dist = sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
                return dist < 1.1*R

        bigcircle().mark(self.problem.cells_meshfunction, 1)
        self.dx = dx(subdomain_data=self.problem.cells_meshfunction)

        bc1 = DirichletBC(self.V_theta, Constant([1.0, 0.0]), neartip)
        bc2 = DirichletBC(self.V_theta, Constant([0.0, 0.0]), outside)
        bc = [bc1, bc2]
        a = inner(grad(self.theta_trial), grad(self.theta_test))*dx
        L = inner(Constant([0.0, 0.0]), self.theta_test)*dx
        solve(a == L, self.theta, bc, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})

    def calc_gtheta(self):

        sigma = self.problem.materials[0].sigma(self.problem.u, self.problem.alpha)
        psi = self.problem.materials[0].elastic_energy_density(self.problem.u, self.problem.alpha)
        G = inner(sigma, dot(grad(self.problem.u), grad(self.theta))) - psi*div(self.theta)
        self.G_value = assemble(G*self.dx(1))
