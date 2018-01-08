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

CircleUt = '''
class CircleUt : public Expression
{
public:

  CircleUt() : Expression(2), mu(1.0), kappa(3.0), t(0.0) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    double r = sqrt(x[0]*x[0]+x[1]*x[1]);
    double theta = atan2(x[1], x[0]);

    values[0] = -t/(8*mu)*(-r*(kappa-3)*cos(theta)+2/r*(-(1-kappa)*cos(theta)+cos(3*theta))-2/pow(r, 3)*cos(3*theta));
    values[1] = t/(8*mu)*(r*(kappa+1)*sin(theta)+2/r*((1+kappa)*sin(theta)-sin(3*theta))+2/pow(r, 3)*sin(3*theta));
  }

  double mu, kappa, t;

};'''

EllipticUt = '''
#include <cmath>
#include <complex>
class EllipticUt : public Expression
{
public:

  EllipticUt() : Expression(2), rho(0.5), mu(1.0), kappa(3.0), t(0.0) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    // Elliptic hole geometry
    double a = 1;
    double b = rho*a;
    double c = sqrt(pow(a, 2)-pow(b, 2));
    double xi0 = std::acosh(a/c);

    // Conversion from cartesian to elliptic
    std::complex<double> i(0, 1);
    std::complex<double> zeta = std::acosh((x[0]+x[1]*i)/c);
    double xi = zeta.real();
    double eta = zeta.imag();

    // Scale factors
    double h = c*sqrt(pow(sinh(xi), 2)+pow(sin(eta), 2));
    double xihat1 = c*sinh(xi)*cos(eta)/h;
    double xihat2 = c*cosh(xi)*sin(eta)/h;
    double etahat1 = -c*cosh(xi)*sin(eta)/h;
    double etahat2 = c*sinh(xi)*cos(eta)/h;

    // Displacement
    double uxi = sqrt(2)*t/(16*mu)*sqrt(pow(a, 2)-pow(b, 2))/sqrt(cosh(2*xi)-cos(2*eta))*((kappa-1)*cosh(2*xi)-(kappa+1)*cos(2*eta)+2*cosh(2*xi0))+sqrt(2)*t/(16*mu)*sqrt(pow(a, 2)-pow(b, 2))/sqrt(cosh(2*xi)-cos(2*eta))*exp(2*xi0)*((kappa-1)*exp(-2*xi)-(kappa+1)*cos(2*eta)+2*exp(-2*xi0)-2*sinh(2*(xi-xi0))*cos(2*eta));
    double ueta = sqrt(2)*t/(16*mu)*sqrt(pow(a, 2)-pow(b, 2))/sqrt(cosh(2*xi)-cos(2*eta))*exp(2*xi0)*(2*cosh(2*(xi-xi0))*sin(2*eta)+2*(kappa-1)*cos(eta)*sin(eta));

    values[0] = uxi*xihat1 + ueta*etahat1;
    values[1] = uxi*xihat2 + ueta*etahat2;
  }

  double rho, mu, kappa, t;

};'''

class EllipticHole(QuasiStaticGradentDamageProblem):

    def __init__(self, ell, rho=1, L=100):
        self.rho = rho
        self.ell = ell
        self.L = L
        QuasiStaticGradentDamageProblem.__init__(self)
        self.W = FunctionSpace(self.mesh, "DG", self.parameters.fem.u_degree-1)
        self.sig, self.sig_trial, self.sig_test = Function(self.W, name="Stress"), TrialFunction(self.W), TestFunction(self.W)
        self.f_sig = XDMFFile(mpi_comm_world(), self.save_dir + "sig.xdmf")
        self.rayleigh = []

    def prefix(self):
        return "hole"

    def set_user_parameters(self):
        p = self.parameters

        p.time.min = 0.0
        p.time.max = 1.0
        p.time.nsteps = 100

        p.problem.stability = False

        p.problem.hsize = self.ell/10  # 1e-3
        p.problem.add("rho", self.rho)
        p.problem.add("L", self.L)

        p.material.ell = self.ell
        p.material.law = "AT1"
        p.material.E = 1.0
        p.material.Gc = 8*self.ell/3
        p.material.nu = 0.3
        p.material.pstress = True

        p.AM.max_iterations = 1000

        p.post_processing.save_energies = True
        p.post_processing.save_u = True
        p.post_processing.save_alpha = True
        p.post_processing.add("save_sig22", True)

        p.solver_u.linear_solver = "cg"
        p.solver_u.preconditioner = "hypre_amg"
        p.solver_alpha.method = "tron"

    def define_mesh(self):
        geofile = \
            """
            Mesh.RandomFactor = 1e-10;
            rho = DefineNumber[ %g, Name "Parameters/rho" ];
            L = DefineNumber[ %g, Name "Parameters/L" ];
            lc = DefineNumber[ %g, Name "Parameters/lc" ];
            lc1 = DefineNumber[ rho*rho*lc, Name "Parameters/lc1" ];
            lc2 = DefineNumber[ lc/rho, Name "Parameters/lc2" ];
            lc3 = DefineNumber[ L/100, Name "Parameters/lc3" ];
            Point(1) = {0, 0, 0, lc1};
            Point(2) = {1, 0, 0, lc1};
            Point(3) = {L, 0, 0, lc3};
            Point(4) = {0, rho, 0, lc2};
            Point(5) = {0, L, 0, lc3};
            Ellipse(1) = {2, 1, 2, 4};
            Ellipse(2) = {3, 1, 3, 5};
            Line(3) = {5, 4};
            Line(4) = {2, 3};
            Line Loop(1) = {3, -1, 4, 2};
            Plane Surface(1) = {1};
            """ % (self.rho, self.L, self.parameters.problem.hsize)

        if self.rho == 1:
            rho_str = "1"
        else:
            rho_str = str(self.rho).replace(".", "x")

        return mesher(geofile, "hole_rho%s_L%g" %(rho_str, self.L))

    def set_user_job(self):

        # Stop alternate minimization as soon as one finds a broken state
        alpha_max = self.alpha.vector().max()
        if alpha_max > 0.99:
            self.user_break = True

    def set_user_post_processing(self):

        # Project the sig
        if self.parameters.post_processing.save_sig22:
            sigma = self.materials[0].sigma(self.u, self.alpha)
            a = inner(self.sig_trial, self.sig_test)*dx
            L = inner(sigma[1, 1], self.sig_test)*dx
            projector = LocalSolver(a, L)
            projector.solve_local_rhs(self.sig)
            self.f_sig.write(self.sig, self.t)

        # Save the Rayleigh quotient evolution
        if self.parameters.problem.stability:
            self.rayleigh.append([self.t, self.rq])
            pl.savetxt(self.save_dir + "rayleigh.txt", pl.array(self.rayleigh), "%.3e")

        if self.user_break:
            self.file_alpha << (self.alpha, self.t)

    def set_mesh_functions(self):
        self.cells_meshfunction = MeshFunction("size_t", self.mesh, self.dimension)
        self.cells_meshfunction.set_all(0)
        self.exterior_facets_meshfunction = MeshFunction("size_t", self.mesh, self.dimension - 1)
        self.exterior_facets_meshfunction.set_all(0)

        class Left(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0.)

        class Down(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], 0.)

        L = self.L
        h = L/100
        class Ext(SubDomain):
            def inside(self, x, on_boundary):
                return sqrt(x[0]*x[0] + x[1]*x[1]) > L-h/2.0

        Left().mark(self.exterior_facets_meshfunction, 1)
        Down().mark(self.exterior_facets_meshfunction, 2)
        Ext().mark(self.exterior_facets_meshfunction, 3)

    def define_bc_u(self):

        mu = float(self.materials[0].mu)
        nu = float(self.materials[0].nu)
        kappa = (3-nu)/(1+nu)

        if self.rho == 1:
            Ut = Expression(CircleUt, degree=3)
        else:
            Ut = Expression(EllipticUt, degree=3)
            Ut.rho = self.rho
        Ut.mu = mu; Ut.kappa = kappa; Ut.t = 0.0

        bc_1 = DirichletBC(self.V_u.sub(0), Constant(0.0), self.exterior_facets_meshfunction, 1)
        bc_2 = DirichletBC(self.V_u.sub(1), Constant(0.0), self.exterior_facets_meshfunction, 2)
        bc_3 = DirichletBC(self.V_u, Ut, self.exterior_facets_meshfunction, 3)
        return [bc_1, bc_2, bc_3]

if __name__ == '__main__':

    # Run a simulation without bifurcation analysis
    problem = EllipticHole(rho=0.1, ell=0.1)
    problem.solve()
