# Copyright (C) 2014 Tianyi Li
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

# Problem
class Surfing(QuasiStaticGradentDamageProblem):

    def __init__(self, hsize, ell, law="AT1", k=2.0):
        self.hsize = hsize
        self.ell = ell
        self.law = law
        self.k = k
        QuasiStaticGradentDamageProblem.__init__(self)

        # For g-theta calculation
        self.Gtheta = Gtheta(self)
        self.Gtheta_history = []

    def prefix(self):
        return "surfing"

    def set_user_parameters(self):

        p = self.parameters

        p.problem.hsize = self.hsize

        p.time.min = 0.0
        p.time.max = 1.0
        p.time.nsteps = 100

        p.material.Gc = 1.5
        p.material.nu = 0.2
        p.material.ell = self.ell
        p.material.law = self.law
        p.material.k = self.k
        p.material.pstress = True

        p.AM.max_iterations = 1000
        p.AM.tolerance = 1e-5

        p.post_processing.save_energies = True
        p.post_processing.save_alpha = True
        p.post_processing.save_u = True
        p.post_processing.plot_alpha = True

        p.solver_u.linear_solver = "gmres"
        p.solver_u.preconditioner = "hypre_amg"

        p.solver_alpha.method = "tron"
        p.solver_alpha.linear_solver = "stcg"
        p.solver_alpha.preconditioner = "hypre_amg"

    def define_mesh(self):
        geofile = \
            """
            h = DefineNumber[ %g, Name "Parameters/h" ];
            b = DefineNumber[ %g, Name "Parameters/b" ];
            Point(1) = {0, -0.5, 0, 0.05};
            Point(2) = {5, -0.5, 0, 0.05};
            Point(3) = {5, 0.5, 0, 0.05};
            Point(4) = {0, 0.5, 0, 0.05};
            Point(5) = {0, -b, 0, h};
            Point(6) = {5, -b, 0, h};
            Point(7) = {5, b, 0, h};
            Point(8) = {0, b, 0, h};
            Point(9) = {0, 0, 0, h};
            Point(10) = {1, 0, 0, h};
            Line(1) = {1, 2};
            Line(2) = {2, 6};
            Line(3) = {6, 5};
            Line(4) = {5, 1};
            Line(5) = {8, 7};
            Line(6) = {7, 3};
            Line(7) = {3, 4};
            Line(8) = {4, 8};
            Line(9) = {6, 7};
            Line(10) = {8, 9};
            Line(11) = {9, 5};
            Line(12) = {9, 10};
            Line Loop(1) = {1, 2, 3, 4};
            Line Loop(2) = {5, 6, 7, 8};
            Line Loop(3) = {-3, 9, -5, 10, 11};
            Plane Surface(1) = {1};
            Plane Surface(2) = {2};
            Plane Surface(3) = {3};
            Line {12} In Surface {3};
            Physical Surface(0) = {1, 2, 3};
            """ %(self.hsize, min(20*self.hsize, 0.4))
        return mesher(geofile, "surfing_h%g" %(self.hsize))

    def define_initial_alpha(self):
        return Expression("x[1] == 0 & x[0] <= 1.0 ? 1.0 : 0.0", degree=1)

    def define_bc_u(self):

        mu = float(self.materials[0].mu)
        nu = float(self.materials[0].nu)
        kappa = (3.0-nu)/(1.0+nu)
        KI = 1.0
        v = 4.0

        # Surfing displacement
        U = Expression(["KI/(2*mu)*sqrt(sqrt((x[0]-1-v*t)*(x[0]-1-v*t)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0]-1-v*t)))*cos(atan2(x[1], x[0]-1-v*t)/2)",
                        "KI/(2*mu)*sqrt(sqrt((x[0]-1-v*t)*(x[0]-1-v*t)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0]-1-v*t)))*sin(atan2(x[1], x[0]-1-v*t)/2)"],
                        KI=KI, mu=mu, kappa=kappa, v=v, t=0.0, degree=2)

        bc = DirichletBC(self.V_u, U, "on_boundary")
        return [bc]

    def set_user_post_processing(self):
        if self.parameters.material.law == "AT1":
            Gc_num = (1+3*self.hsize/(8*self.ell)) * self.parameters.material.Gc
        elif self.parameters.material.law == "AT2":
            Gc_num = (1+self.hsize/(2*self.ell)) * self.parameters.material.Gc
        elif self.parameters.material.law == "ATk":
            Gc_num = (1+self.hsize/(pl.pi*self.ell)) * self.parameters.material.Gc
        x0 = self.dissipated_energy_value / Gc_num  # estimation of the current crack tip location

        # Calculate the energy release rate
        self.Gtheta.x0 = x0
        self.Gtheta.r = 2*self.ell
        if self.parameters.material.law == "ATk":
            self.Gtheta.r = min(2*self.parameters.material.k*self.ell, 0.45)
        self.Gtheta.R = 2.5*self.Gtheta.r

        self.Gtheta.calc_theta()
        self.Gtheta.f_theta.write(self.Gtheta.theta, self.t)

        self.Gtheta.calc_gtheta()
        self.Gtheta_history.append(self.Gtheta.G_value)
        self.print0("    gtheta = %.3e" %(self.Gtheta.G_value))
        pl.savetxt(self.save_dir + "gtheta.txt", pl.array(self.Gtheta_history), "%.5e")

if __name__ == '__main__':

    # Run a fast simulation
    problem = Surfing(hsize=0.05, ell=0.1, law="AT1")
    problem.solve()
