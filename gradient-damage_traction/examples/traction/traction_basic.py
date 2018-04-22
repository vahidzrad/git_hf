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

    def __init__(self, ndim, hsize):
        self.hsize = hsize
        self.ndim = ndim
        QuasiStaticGradentDamageProblem.__init__(self)

    def prefix(self):
        return "traction" + str(self.ndim) + "d"

    def set_user_parameters(self):
        p = self.parameters
        p.time.max = 1.1
        p.time.nsteps = 5
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
        geofile = \
		"""
		lc = DefineNumber[ %g, Name "Parameters/lc" ];
                Point(1) = {0, 0, 0, 10*lc};
                Point(2) = {4, 0, 0, 10*lc};
                Point(3) = {4, 4, 0, 10*lc};
                Point(4) = {0, 4, 0, 10*lc};
                Point(5) = {1.8, 2., 0, 1*lc};
                Point(6) = {2.2, 2., 0, 1*lc};
                Line(1) = {1, 2};
                Line(2) = {2, 3};
                Line(3) = {3, 4};
                Line(4) = {4, 1};
                Line Loop(5) = {1, 2, 3, 4};
		Plane Surface(30) = {5};

		Line(6) = {5, 6};
                Line{6} In Surface{30};


		Physical Surface(1) = {30};

		Physical Line(101) = {6};

          	"""%(self.parameters.problem.hsize)

        # Generate XML files
        return mesher(geofile, "fracking_hsize%g" % (self.hsize))




    def set_mesh_functions(self):
        self.cells_meshfunction = MeshFunction("size_t", self.mesh, self.dimension)
        self.cells_meshfunction.set_all(0)
        self.exterior_facets_meshfunction = MeshFunction("size_t", self.mesh, self.dimension - 1)
        self.exterior_facets_meshfunction.set_all(0)

        class Top(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], 4.)

        class Down(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], 0.)

     	class Right(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0.)

      	class Left(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 4.)

        class Crack(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], 2) and  x[0] <= 2.2 and x[0] >= 1.8 and on_boundary




        Top().mark(self.exterior_facets_meshfunction, 1)
        Down().mark(self.exterior_facets_meshfunction, 2)
        Right().mark(self.exterior_facets_meshfunction, 3)
        Left().mark(self.exterior_facets_meshfunction, 4)
        Crack().mark(self.exterior_facets_meshfunction, 5)




    def define_initial_alpha(self):
       return Expression("x[1] == 2. & x[0] <= 2.2 & x[0] >=1.8 ? 1.0 : 0.0", degree=1)


    def define_bc_u(self):
        value0 = ["0"] * self.dimension
        valuet = ["0"] * self.dimension
        valueb = ["0"] * self.dimension

        valuet[0] = "1*t"
        valueb[0] = "-1*t"
        bc_1 = DirichletBC(self.V_u, Expression(valuet, t=0.0, degree=1), self.exterior_facets_meshfunction, 1)
        bc_2 = DirichletBC(self.V_u, Expression(valueb, t=0.0, degree=1), self.exterior_facets_meshfunction, 2)
        return [bc_1, bc_2]

    def define_bc_alpha(self):
        bc_1 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 1)
        bc_2 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 2)
        bc_3 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 3)
        bc_4 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 4)
        bc_5 = DirichletBC(self.V_alpha, Constant(1.0), self.exterior_facets_meshfunction, 5)
        return [bc_1, bc_2, bc_3, bc_4, bc_5]

if __name__ == "__main__":

    # Run a 2-d simulation
    problem = Traction(ndim=2, hsize=0.1)
    problem.solve()
