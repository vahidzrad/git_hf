# Copyright (C) 2017 Tianyi Li, Corrado Maurini
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

class Initiation(QuasiStaticGradentDamageProblem):

    def __init__(self, desc="crack"):

        # Parameters
        self.desc = desc

        # Basic initialisation
        QuasiStaticGradentDamageProblem.__init__(self)

        # Initialisation for gtheta and Rayleigh
        self.Gtheta = []
        self.f_theta = File(self.save_dir + "theta.pvd")
        self.Rayleigh = []

        # To estimate the current crack tip by finding the maximum x coordinate where alpha > X
        self.d2v = dof_to_vertex_map(self.V_alpha)
        self.xcoor = self.mesh.coordinates()[self.d2v][:, 0]

    def prefix(self):
        return "initiation"

    def set_user_parameters(self):

        p = self.parameters

        p.AM.max_iterations = 1000
        p.AM.tolerance = 1e-5

        p.time.min = 0.0
        p.time.max = 1.7
        p.time.nsteps = 50

        p.fem.u_degree = 1

        p.material.Gc = 1.0
        p.material.nu = 0.3
        p.material.ell = 0.05
        p.material.pstress = True

        p.problem.hsize = p.material.ell/10.0
        p.problem.add("desc", self.desc)
        p.problem.stability = True

        p.post_processing.save_energies = True
        p.post_processing.save_alpha = True
        p.post_processing.save_u = True
        p.post_processing.plot_alpha = True

        p.solver_u.linear_solver = "cg"
        p.solver_u.preconditioner = "hypre_amg"

        p.solver_alpha.method = "gpcg"
        p.solver_alpha.linear_solver = "nash"
        p.solver_alpha.preconditioner = "bjacobi"

    def define_mesh(self):

        if self.desc == "crack":
            geofile = \
                """
                lc = DefineNumber[ %g, Name "Parameters/lc" ];
                lc2 = DefineNumber[ 5*lc, Name "Parameters/lc2" ];
                Point(1) = {-0.5, -0.5, 0, lc2};
                Point(2) = {0.5, -0.5, 0, lc2};
                Point(3) = {0.5, 0.5, 0, lc2};
                Point(4) = {-0.5, 0.5, 0, lc2};
                Point(5) = {0, 0, 0, lc};
                Point(6) = {-0.5, 1e-6, 0, lc2};
                Point(7) = {0.5, 0, 0, lc2};
                Point(8) = {-0.5, -1e-6, 0, lc2};
                Line(1) = {1, 2};
                Line(2) = {2, 7};
                Line(3) = {7, 3};
                Line(4) = {3, 4};
                Line(5) = {4, 6};
                Line(6) = {8, 1};
                Line(7) = {6, 5};
                Line(8) = {5, 7};
                Line(9) = {5, 8};
                Line Loop(1) = {1, 2, -8, 9, 6};
                Plane Surface(1) = {1};
                Line Loop(2) = {7, 8, 3, 4, 5};
                Plane Surface(2) = {2};
                """ %(self.parameters.problem.hsize)

        elif self.desc == "damage":
            geofile = \
                """
                lc = DefineNumber[ %g, Name "Parameters/lc" ];
                lc2 = DefineNumber[ 5*lc, Name "Parameters/lc2" ];
                Point(1) = {-0.5, -0.5, 0, lc2};
                Point(2) = {0.5, -0.5, 0, lc2};
                Point(3) = {0.5, 0.5, 0, lc2};
                Point(4) = {-0.5, 0.5, 0, lc2};
                Point(5) = {-0.5, 0, 0, lc};
                Point(6) = {0, 0, 0, lc};
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

        return mesher(geofile, "initiation_%s" %(self.desc))

    def define_initial_alpha(self):

        if self.desc == "crack":
            return Constant(0.0)
        elif self.desc == "damage":
            return Expression("fabs(x[1]) <= %g & x[0] <= 0.0 ? 1.0 : 0.0" % (1.5*self.parameters.problem.hsize))

    def define_bc_u(self):

        mu = float(self.materials[0].mu)
        nu = float(self.materials[0].nu)
        kappa = (3.0-nu)/(1.0+nu)

        # Singular displacement
        Ut = Expression(["sqrt(t)/(2*mu)*sqrt(sqrt(x[0]*x[0]+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0])))*cos(atan2(x[1], x[0])/2)", "sqrt(t)/(2*mu)*sqrt(sqrt(x[0]*x[0]+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0])))*sin(atan2(x[1], x[0])/2)"], mu=mu, kappa=kappa, t=0.0, degree=1)

        bc = DirichletBC(self.V_u, Ut, "near(x[1], 0.5) || near(x[1], -0.5) || near(x[0], 0.5) || near(x[0], -0.5)")
        return [bc]

    def set_user_post_processing(self):

        # Estimate the current crack tip
        ind = self.alpha.vector().array() > 0.5
        if ind.any():
            xmax = self.xcoor[ind].max()
        else:
            xmax = 0.0
        x0 = MPI.max(mpi_comm_world(), xmax)

        # Calculate G
        gtheta = Gtheta(self)
        gtheta.x0 = x0
        gtheta.r = 2*self.parameters.material.ell
        gtheta.R = 2.5*gtheta.r
        gtheta.calc_theta()
        gtheta.calc_gtheta()
        self.f_theta << (gtheta.theta, self.t)
        self.Gtheta.append([self.t, gtheta.G_value])
        self.print0("Gtheta equals %.3e" %(gtheta.G_value))
        pl.savetxt(self.save_dir + "Gtheta.txt", pl.array(self.Gtheta), "%.5e")

        # Calculate Rayleigh
        self.Rayleigh.append([self.t, self.rq])
        pl.savetxt(self.save_dir + "Rayleigh.txt", pl.array(self.Rayleigh), "%.5e")

if __name__ == '__main__':

    problem = Initiation(desc="crack")
    problem.solve()
