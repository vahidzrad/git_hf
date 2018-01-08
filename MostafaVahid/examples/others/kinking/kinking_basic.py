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

class Kinking(QuasiStaticGradentDamageProblem):

    def __init__(self, K2=1.0, nu=0.0, ell=0.1):

        # Parameters
        self.K2 = K2
        self.nu = nu
        self.ell = ell

        # Initialisation
        QuasiStaticGradentDamageProblem.__init__(self)

    def prefix(self):
        return "kinking"

    def set_user_parameters(self):

        p = self.parameters

        p.problem.hsize = self.ell/5
        p.problem.add("K2", self.K2)

        p.AM.max_iterations = 1000
        p.AM.tolerance = 1e-5

        p.time.min = 0.0
        p.time.max = 1.0
        p.time.nsteps = 2

        p.material.Gc = 1.0
        p.material.nu = self.nu
        p.material.ell = self.ell

        p.post_processing.save_energies = False
        p.post_processing.save_alpha = True
        p.post_processing.save_u = True
        p.post_processing.plot_alpha = True

        p.solver_u.linear_solver = "gmres"
        p.solver_u.preconditioner = "bjacobi"

        p.solver_alpha.method = "gpcg"
        p.solver_alpha.linear_solver = "nash"
        p.solver_alpha.preconditioner = "bjacobi"

    def define_materials(self):

        class MyGradientDamageMaterial(GradientDamageMaterial):

            def __init__(self, material_parameters):
                GradientDamageMaterial.__init__(self, material_parameters)
                self.lmbda = self.E*self.nu/(1.0-self.nu**2)  # plane stress condition

        return [MyGradientDamageMaterial(self.parameters.material)]

    def define_mesh(self):
        # geofile = \
        # """
        # h = DefineNumber[ %g, Name "Parameters/h" ];
        # Point(1) = {-0.5, -0.5, 0, 2*h};
        # Point(2) = {0.5, -0.5, 0, h};
        # Point(3) = {0.5, 0.5, 0, 2*h};
        # Point(4) = {-0.5, 0.5, 0, 5*h};
        # Point(5) = {-0.5, h, 0, 5*h};
        # Point(6) = {-h, h, 0, h};
        # Point(7) = {0, 0, 0, h};
        # Point(8) = {-h, -h, 0, h};
        # Point(9) = {-0.5, -h, 0, 5*h};
        # Point(10) = {0.25, -0.25, 0, h};
        # Line(1) = {1, 2};
        # Line(2) = {2, 3};
        # Line(3) = {3, 4};
        # Line(4) = {4, 5};
        # Line(5) = {5, 6};
        # Line(6) = {6, 7};
        # Line(7) = {7, 8};
        # Line(8) = {8, 9};
        # Line(9) = {9, 1};
        # Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        # Plane Surface(1) = {1};
        # Point{10} In Surface{1};
        # """ % (self.parameters.problem.hsize)

        geofile = \
            """
            h = DefineNumber[ %g, Name "Parameters/h" ];
            Point(1) = {-0.5, -0.5, 0, 2*h};
            Point(2) = {0.5, -0.5, 0, h};
            Point(3) = {0.5, 0.5, 0, 2*h};
            Point(4) = {-0.5, 0.5, 0, 5*h};
            Point(5) = {-0.5, 0, 0, h};
            Point(6) = {0, 0, 0, h};
            Line(1) = {1, 2};
            Line(2) = {2, 3};
            Line(3) = {3, 4};
            Line(4) = {4, 5};
            Line(5) = {5, 1};
            Line(6) = {5, 6};
            Line Loop(1) = {1, 2, 3, 4, 5};
            Plane Surface(1) = {1};
            Line{6} In Surface{1};
            """ % (self.parameters.problem.hsize)

        return mesher(geofile, "kinking_h%g_alpha0" % (self.parameters.problem.hsize))

    def define_initial_alpha(self):
        # return Constant(0.0)
        return Expression("x[1] == 0 & x[0] <= 0.0 ? 1.0 : 0.0", degree=1)

    def define_bc_u(self):

        E = float(self.materials[0].E)
        mu = float(self.materials[0].mu)
        nu = float(self.materials[0].nu)
        kappa = (3.0-nu)/(1.0+nu)
        K1 = sqrt(1.5)/sqrt(1+self.K2**2)
        K2 = self.K2*K1

        # Singular displacement
        Ut = Expression(["t*K1/(2*mu)*sqrt(sqrt(x[0]*x[0]+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0])))*cos(atan2(x[1], x[0])/2)+\
                          t*K2/E*sqrt(sqrt(x[0]*x[0]+x[1]*x[1])/(2*pi))*(1+nu)*(2+kappa+cos(atan2(x[1], x[0])))*sin(atan2(x[1], x[0])/2)",
                         "t*K1/(2*mu)*sqrt(sqrt(x[0]*x[0]+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0])))*sin(atan2(x[1], x[0])/2)+\
                          t*K2/E*sqrt(sqrt(x[0]*x[0]+x[1]*x[1])/(2*pi))*(1+nu)*(2-kappa-cos(atan2(x[1], x[0])))*cos(atan2(x[1], x[0])/2)"],
                          K1=K1, K2=K2, E=E, mu=mu, nu=nu, kappa=kappa, t=0.0, degree=3)

        bc = DirichletBC(self.V_u, Ut, "on_boundary")
        return [bc]

if __name__ == '__main__':

    # Run a fast simulation
    problem = Kinking(ell=0.5)
    problem.solve()
