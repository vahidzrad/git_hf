# Copyright (C) 2014 Tianyi Li
# Licensed under the GNU LGPL Version 3.
from gradient_damage import *

set_log_level(ERROR)

# Problem
class Fracking(QuasiStaticGradentDamageProblem):

    def __init__(self, hsize, ell):

        # Parameters
        self.hsize = hsize
        self.ell = ell


        # Initialisation
        QuasiStaticGradentDamageProblem.__init__(self)

    def prefix(self):
        return "hydraulicfracturing"

    def set_user_parameters(self):
        p = self.parameters
        p.time.max = 1.0
        #p.material.ell = 0.1
        #p.material.Gc = 8*p.material.ell/3
        p.material.Gc = 1.0
        p.material.E = 1.0
        p.material.nu = 0.0
        p.material.law = "AT1"
        p.post_processing.plot_alpha = True
        p.post_processing.plot_u = False
        p.post_processing.save_alpha = True
        p.post_processing.save_u = True
        p.post_processing.save_Beta = False
        p.post_processing.save_V = False
        p.solver_alpha.method = "gpcg"


    def define_mesh(self):
        geofile = \
		"""
		lc = DefineNumber[ %g, Name "Parameters/lc" ];
                Point(1) = {0, 0, 0, 100*lc};
                Point(2) = {4, 0, 0, 100*lc};
                Point(3) = {4, 4, 0, 100*lc};
                Point(4) = {0, 4, 0, 100*lc};
                Point(5) = {1.8, 2., 0, 10*lc};
                Point(6) = {2.2, 2., 0, 10*lc};
                Line(1) = {1, 2};
                Line(2) = {2, 3};
                Line(3) = {3, 4};
                Line(4) = {4, 1};
                Line Loop(5) = {1, 2, 3, 4};
		Plane Surface(1) = {5};
		Line(6) = {5, 6};
                Line{6} In Surface{1};
		Physical Line(22) = {6};
		Physical Surface(1) = {1};
            """%(self.parameters.problem.hsize)

        # Generate XML files
        return mesher(geofile, "fracking_hsize%g" % (self.hsize))

    def define_materials(self):

        class MyGradientDamageMaterial(GradientDamageMaterial):

            def P_b(self):
    	   	return Expression("t", t=0.0, degree =1)

        material1 = MyGradientDamageMaterial(self.parameters.material, "everywhere")
        return [material1]


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

        valuet[0] = "5*t"
        valueb[0] = "-5*t"
        bc_1 = DirichletBC(self.V_u, Expression(valuet, t=0.0, degree=1), self.exterior_facets_meshfunction, 1)
        bc_2 = DirichletBC(self.V_u, Expression(valueb, t=0.0, degree=1), self.exterior_facets_meshfunction, 2)
        return []

    def define_bc_alpha(self):
        bc_1 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 1)
        bc_2 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 2)
        bc_3 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 3)
        bc_4 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 4)
        bc_5 = DirichletBC(self.V_alpha, Constant(1.0), self.exterior_facets_meshfunction, 5)
        return [bc_1, bc_2, bc_3, bc_4, bc_5]



		
if __name__ == '__main__':

    # Run a fast simulation
    problem = Fracking(hsize=0.1, ell=1.0e-2)
    problem.solve()



