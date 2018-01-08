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

from gradient_damage import *

set_log_level(ERROR)

class Traction(QuasiStaticGradentDamageProblem):

    def __init__(self, ndim):
        self.ndim = ndim
        QuasiStaticGradentDamageProblem.__init__(self)

    def prefix(self):
        return "traction" + str(self.ndim) + "d"

    def set_user_parameters(self):
        p = self.parameters
        p.problem.stability = True  # activate bifurcation analysis
        p.problem.stability_correction = True  # ensure that the solution is always unique
        p.time.max = 1.1
        p.material.ell = 0.1
        p.material.Gc = 8*p.material.ell/3
        p.material.law = "AT1"
        p.post_processing.plot_alpha = False
        p.post_processing.save_alpha = True
        p.post_processing.save_u = False
        p.post_processing.save_Beta = False
        p.post_processing.save_V = False
        p.solver_alpha.method = "gpcg"

    def define_mesh(self):
        if self.ndim == 1:
            mesh = UnitIntervalMesh(100)
        elif self.ndim == 2 :
            mesh = RectangleMesh(Point(0., 0.), Point(1., .1), 100, 10)
        elif self.ndim == 3 :
            mesh = BoxMesh(Point(0., 0., 0.), Point(1., .1, .1), 100, 10, 10)
        return mesh

    def define_bc_u(self):
        value0 = ["0"] * self.dimension
        valuet = ["0"] * self.dimension
        valuet[0] = "t"
        bc_1 = DirichletBC(self.V_u, Constant(value0), "near(x[0], 0)")
        bc_2 = DirichletBC(self.V_u, Expression(valuet, t=0.0, degree=1), "near(x[0], 1)")
        return [bc_1, bc_2]

if __name__ == "__main__":

    # Run a 2-d simulation
    problem = Traction(2)
    problem.solve()
