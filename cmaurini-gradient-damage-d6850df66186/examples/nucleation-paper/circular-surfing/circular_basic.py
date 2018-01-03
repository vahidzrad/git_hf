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

from gradient_damage import *
from circular_arc_crack_solution import *

set_log_level(ERROR)

# Specific gtheta class for circular surfing
class GthetaCircular(Gtheta):

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

        bc1 = DirichletBC(self.V_theta, Constant([1.0, 1.0]), neartip)
        bc2 = DirichletBC(self.V_theta, Constant([0.0, 0.0]), outside)
        bc = [bc1, bc2]
        a = inner(grad(self.theta_trial), grad(self.theta_test))*dx
        L = inner(Constant([0.0, 0.0]), self.theta_test)*dx
        solve(a == L, self.theta, bc, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
        n = interpolate(Expression(["-x[1]/sqrt(x[0]*x[0]+x[1]*x[1])", "+x[0]/sqrt(x[0]*x[0]+x[1]*x[1])"], degree=2), self.problem.V_u)
        self.theta.vector()[:] = n.vector()*self.theta.vector()

# Problem
class Circular(QuasiStaticGradentDamageProblem):

    def __init__(self):

        # Initialisation
        QuasiStaticGradentDamageProblem.__init__(self)

        # For g-theta calculation
        self.Gtheta = GthetaCircular(self)
        self.Gtheta_history = []

    def prefix(self):
        return "circular"

    def set_user_parameters(self):

        p = self.parameters

        p.problem.hsize = 0.004

        p.time.min = pl.pi/8
        p.time.max = pl.pi/2
        p.time.nsteps = 100

        p.material.E = 1.0
        p.material.Gc = 1.0
        p.material.nu = 0.0
        p.material.ell = 0.02
        p.material.law = "AT1"

        p.AM.max_iterations = 10000
        p.AM.tolerance = 1e-5

        p.post_processing.save_energies = True
        p.post_processing.save_alpha = True
        p.post_processing.save_u = True
        p.post_processing.plot_alpha = True

        p.solver_u.linear_solver = "cg"
        p.solver_u.preconditioner = "hypre_amg"

    def define_mesh(self):
        geofile = \
            """
            h = DefineNumber[ 0.004, Name "Parameters/h" ];
            a = DefineNumber[ 0.2, Name "Parameters/a" ];
            R = DefineNumber[ 1, Name "Parameters/R" ];
            theta0 = DefineNumber[ Pi/8, Name "Parameters/theta0" ];
            Point(1) = {0, 0, 0, h};
            Point(2) = {R-a, 0, 0, h};
            Point(3) = {R+a, 0, 0, h};
            Point(4) = {0, R-a, 0, h};
            Point(5) = {0, R+a, 0, h};
            Point(6) = {R*Cos(theta0), R*Sin(theta0), 0, h};
            Point(7) = {R, 0, 0, h};

            Circle(6) = {7, 1, 6};
            Circle(7) = {2, 1, 4};
            Circle(8) = {3, 1, 5};
            Line(9) = {5, 4};
            Line(10) = {2, 7};
            Line(11) = {7, 3};

            Line Loop(1) = {7, -9, -8, -11, -10};
            Plane Surface(1) = {1};

            Line{6} In Surface{1};
            Physical Surface(0) = {1};
            """

        return mesher(geofile, "circular")

    def define_initial_alpha(self):
        return Expression("near(x[0]*x[0]+x[1]*x[1], 1) && x[1] <= sin(theta0) ? 1.0 : 0.0", theta0=pl.pi/8, degree=2)

    def define_bc_u(self):

        mu = float(self.materials[0].mu)
        nu = float(self.materials[0].nu)
        kappa = 3.0-4.0*nu
        R = 1.0

        class Muskhelishvili(Expression):

            def __init__(self, degree):
                self.t = 0.0

            def eval(self, value, x):
                pyy = (15+20*pl.cos(self.t)-3*pl.cos(2*self.t))/(16*pl.sqrt(pl.pi*R*pl.sin(self.t))*pl.cos(self.t/2))
                pxx = pyy*(16/(3*(pl.cos(2*self.t)-5)-20*pl.cos(self.t))+1)
                displ, stress, KI, KII = get_circular_arc_crack_solution(self.t, R, pxx, pyy, 0, mu, kappa)
                U = displ(x[0]+x[1]*1j)

                value[0] = U[0]
                value[1] = U[1]

            def value_shape(self):
                return (2,)

        U = Muskhelishvili(degree=3)
        bc = DirichletBC(self.V_u, U, "on_boundary")
        return [bc]

    def set_user_post_processing(self):

        # Calculate the energy release rate
        self.Gtheta.x0 = cos(self.t)
        self.Gtheta.y0 = sin(self.t)
        self.Gtheta.r = 2*self.parameters.material.ell
        self.Gtheta.R = 2.5*self.Gtheta.r

        self.Gtheta.calc_theta()
        self.Gtheta.f_theta.write(self.Gtheta.theta, self.t)

        self.Gtheta.calc_gtheta()
        self.Gtheta_history.append(self.Gtheta.G_value)
        self.print0("    gtheta = %.3e" %(self.Gtheta.G_value))
        pl.savetxt(self.save_dir + "gtheta.txt", pl.array(self.Gtheta_history), "%.5e")

if __name__ == '__main__':

    problem = Circular()
    problem.solve()
