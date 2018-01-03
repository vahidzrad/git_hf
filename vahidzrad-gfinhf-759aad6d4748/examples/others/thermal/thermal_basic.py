# Copyright (C) 2014 Tianyi Li
# Licensed under the GNU LGPL Version 3.
from gradient_damage import *

set_log_level(ERROR)

# Problem
class Thermal(QuasiStaticGradentDamageProblem):

    def __init__(self, hsize):

        # Parameters
        self.hsize = hsize

        # Initialisation
        QuasiStaticGradentDamageProblem.__init__(self)

    def prefix(self):
        return "thermal"

    def define_materials(self):

        class MyGradientDamageMaterial(GradientDamageMaterial):

            def __init__(self, material_parameters, subdomain_id):
                self.kappa = 17.73  # 0.47
                self.alpha = 7.7e-6
                self.v = 5
                self.Dtheta = -125
                GradientDamageMaterial.__init__(self, material_parameters, subdomain_id)
                self.lmbda = self.E*self.nu/(1.0-self.nu**2)  # plane stress condition

            def eps0(self):
                eps0 = Expression("x[1] <= v*t ? alpha*Dtheta : alpha*Dtheta*(1-exp(-v*(x[1]-v*t)/kappa))",
                                   alpha=self.alpha, v=self.v, Dtheta=self.Dtheta, kappa=self.kappa, t=0.0, degree =1)
                return eps0

        material1 = MyGradientDamageMaterial(self.parameters.material, "everywhere")
        return [material1]

    def set_user_parameters(self):

        p = self.parameters

        p.problem.hsize = self.hsize

        p.time.min = 0.0
        p.time.max = 10.0
        p.time.nsteps = 100

        p.material.E = 72.3e3
        p.material.Gc = 8.8e-3
        p.material.nu = 0.23
        p.material.ell = 1.0
        p.material.law = "AT1"

        p.AM.max_iterations = 1000
        p.AM.tolerance = 1e-5

        p.post_processing.save_energies = True
        p.post_processing.save_alpha = True
        p.post_processing.save_u = True
        p.post_processing.plot_alpha = True

        p.solver_u.linear_solver = "mumps"
        #p.solver_u.preconditioner = "jacobi"

        p.solver_alpha.method = "gpcg"
        p.solver_alpha.linear_solver = "nash"
        p.solver_alpha.preconditioner = "jacobi"

    def define_mesh(self):
        W = 25
        H = 75
        geofile = \
        """
        h = DefineNumber[ %g, Name "Parameters/h" ];
        W = DefineNumber[ %g, Name "Parameters/W" ];
        H = DefineNumber[ %g, Name "Parameters/H" ];
        Point(1) = {-W/2, 0, 0, h};
        Point(2) = {0, 0, 0, h};
        Point(3) = {W/2, 0, 0, h};
        Point(4) = {W/2, H, 0, h};
        Point(5) = {-W/2, H, 0, h};
        Point(6) = {0, 5, 0, h};
        Line(1) = {1, 2};
        Line(2) = {2, 3};
        Line(3) = {3, 4};
        Line(4) = {4, 5};
        Line(5) = {5, 1};
        Line(6) = {2, 6};
        Line Loop(1) = {1, 2, 3, 4, 5};
        Plane Surface(1) = {1};
        Line {6} In Surface {1};
        Physical Surface(0) = {1};
        """ % (self.hsize, W, H)

        # Generate XML files
        return mesher(geofile, "thermal_hsize%g" % (self.hsize))

    def define_initial_alpha(self):
        return Expression("x[0] == 0 & x[1] <= 5.0 ? 1.0 : 0.0", degree=1)

    def define_bc_u(self):
     #   bc_1 = DirichletBC(self.V_u.sub(0), Constant(0.0), self.exterior_facets_meshfunction, 1)
     #   bc_2 = DirichletBC(self.V_u.sub(0), Expression("t", t=0.0, degree = 0), self.exterior_facets_meshfunction, 2)
        return []

    def define_bc_alpha(self):
    #    bc_1 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 1)
    #    bc_2 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 2)
        return []
if __name__ == '__main__':

    # Run a fast simulation
    problem = Thermal(hsize=0.5)
    problem.solve()