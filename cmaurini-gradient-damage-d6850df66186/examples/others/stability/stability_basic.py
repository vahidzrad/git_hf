# Copyright (C) 2017 Tianyi Li
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

class StabilityBar(QuasiStaticGradentDamageProblem):

    def __init__(self, ndim, ell, law="AT1", k=1.0):
        self.ndim = ndim
        self.ell = ell
        self.law = law
        self.k = k
        QuasiStaticGradentDamageProblem.__init__(self)
        self.rayleigh = []

    def prefix(self):
        return "stability" + str(self.ndim) + "d"

    def set_user_parameters(self):
        p = self.parameters
        p.time.max = 3.0
        p.problem.stability = True
        p.problem.stability_correction = False
        p.material.ell = self.ell
        if self.law == "AT1":
            p.material.Gc = 8*p.material.ell/3
        elif self.law == "ATk":
            p.material.Gc = pl.pi*self.k*self.ell/2
        p.material.law = self.law
        p.material.k = self.k
        p.post_processing.plot_alpha = False
        p.post_processing.save_alpha = True
        p.post_processing.save_u = False
        p.post_processing.save_V = True
        p.post_processing.save_Beta = True
        p.solver_alpha.method = "gpcg"
        # p.solver_slepc.monitor = True
        p.solver_slepc.maximum_iterations = 500

    def define_mesh(self):
        if self.ndim == 1:
            mesh = UnitIntervalMesh(100)
        elif self.ndim == 2 :
            mesh = RectangleMesh(Point(0., 0.), Point(1., .1), 100, 10)
        elif self.ndim == 3 :
            mesh = BoxMesh(Point(0., 0., 0.), Point(1., .1, .1), 100, 10, 10)
        return mesh

    def set_time_stepping(self):
        time = self.parameters.time
        elastic_phase = pl.array([0, 1])
        eps = 1e-4
        damage_phase = pl.linspace(1+eps, time.max, time.nsteps)
        self.time_steps = pl.concatenate([elastic_phase, damage_phase])
        self.user_break = False

    def define_bc_u(self):
        value0 = ["0"] * self.dimension
        valuet = ["0"] * self.dimension
        valuet[0] = "t"
        bc_1 = DirichletBC(self.V_u, Constant(value0), "near(x[0], 0)")
        bc_2 = DirichletBC(self.V_u, Expression(valuet, t=0.0, degree=1), "near(x[0], 1)")
        return [bc_1, bc_2]

    def set_user_post_processing(self):

        # Break if an non-stable solution has been found
        if self.stable:
            U = self.t
        else:
            U = (self.time_steps[self.step-1] + self.time_steps[self.step])/2
            self.user_break = True

        self.rayleigh.append([U, self.rq])
        pl.savetxt(self.save_dir() + "rayleigh.txt", pl.array(self.rayleigh), "%.3e")

if __name__ == '__main__':

    # Run the 2-d simulation
    problem = StabilityBar(1, 0.1)
    problem.solve()
