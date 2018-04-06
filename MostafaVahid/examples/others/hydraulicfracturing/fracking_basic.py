# Copyright (C) 2014 Tianyi Li
# Licensed under the GNU LGPL Version 3.
from gradient_damage import *



# Problem
class Fracking(QuasiStaticGradentDamageProblem):
	
    def __init__(self, hsize, ell, P_constant ):

        # Parameters
        self.hsize = hsize
        self.ell = ell
	self.P_constant = P_constant

        # Initialisation
        QuasiStaticGradentDamageProblem.__init__(self)


    def prefix(self):
        return "hydraulicfracturing"

    def set_user_parameters(self):
        p = self.parameters


        p.time.min = 0.0
        p.time.max = 1.#self.P_constant
        p.time.nsteps = 2

        #p.material.ell = 1e-1
        p.material.Gc = 1/(1+self.hsize/(2*self.ell)) #AT1
        #p.material.Gc = 1/(1+self.hsize/(2*self.ell)) #AT2
        p.material.ell = self.ell


        #p.material.Gc = 1.5e6
        p.material.E = 1.0
        p.material.nu = 0.0
        p.material.law = "AT1"
        p.post_processing.plot_alpha = True
        p.post_processing.plot_u = False
        p.post_processing.save_alpha = True
        p.post_processing.save_u = False
        p.post_processing.save_Beta = False
        p.post_processing.save_V = False
        p.solver_alpha.method = "gpcg"

		#lc = DefineNumber[ %g, Name "Parameters/lc" ];
    def define_mesh(self):
        geofile = \
		"""
		lc = DefineNumber[ %g, Name "Parameters/lc" ];


		r=0.25;
		w=4;
		h=4;
		a=0.2;

                Point(1) = {0, h, 0, 50*lc};
                Point(2) = {w, h, 0, 50*lc};
                Point(3) = {w, 0, 0, 50*lc};
                Point(4) = {0, 0, 0, 50*lc};

                Point(5) = {w/2, h/2, 0, 1*lc};
                Point(6) = {w/2, h/2+r, 0, 1*lc};
                Point(7) = {w/2, h/2-r, 0, 1*lc};

                Point(8) = {w/2, h/2+r+a, 0, 1*lc};
                Point(9) = {w/2, h/2-r-a, 0, 1*lc};

		Line(1) = {1, 2};
		Line(2) = {2, 3};
		Line(3) = {3, 4};
		Line(4) = {4, 1};

		Circle(5) = {6, 5, 7};
		Circle(6) = {7, 5, 6};

		Line(7) = {6, 8};
		Line(8) = {7, 9};

		Line Loop(10) = {1, 2, 3, 4};
		Line Loop(11) = {5, 6};

		Plane Surface(20) = {10, 11};

		Physical Surface(1) = {20};
		Line{7} In Surface{20};
		Line{8} In Surface{20};
		
		Physical Line(21) = {1, 2, 3, 4, 5, 6,7,8};

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

	class Void(SubDomain):
	    def inside(self, x, on_boundary):
		return   x[0] <= 2.5 and x[0] >= 1.5 and  x[1] <= 2.5 and x[1] >= 1.5 and on_boundary
	
        class topCrack(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 2) and  x[1] <= 2.45 and x[0] >= 2.25 and on_boundary

        class bottomCrack(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 2) and  x[1] <= 1.75 and x[0] >= 1.55 and on_boundary


        Top().mark(self.exterior_facets_meshfunction, 1)
        Down().mark(self.exterior_facets_meshfunction, 2)
        Right().mark(self.exterior_facets_meshfunction, 3)
        Left().mark(self.exterior_facets_meshfunction, 4)
        Crack().mark(self.exterior_facets_meshfunction, 5)
	Void().mark(self.exterior_facets_meshfunction, 6)
	topCrack().mark(self.exterior_facets_meshfunction, 7)
	bottomCrack().mark(self.exterior_facets_meshfunction, 8)

    #def P_b(self):
	#P_b=Expression("t", t=0.0, degree =1)
    	#return P_b

    def define_pressure(self):
	pressure = Constant(0.15)
	#pressure = Function(self.V_alpha)
	#input_file_pressure = HDF5File(self.mesh.mpi_comm(), "pressure.h5", "r")
	#input_file_pressure.read(pressure, "solution")
	#input_file_pressure.close()
    	return pressure


    def define_initial_alpha(self):
       #return Expression("x[1] == 2. & x[0] <= 2.2 & x[0] >=1.8 ? 1.0 : 0.0", degree=1)
       #return Expression("x[0] <= 2.5 & x[0] >= 1.5 &  x[1] <= 2.5 & x[1] >= 1.5 ? 1.0 : 0.0", degree=1)
       #return Expression("sqrt( (x[0]-2.)*(x[0]-2.) + (x[1]-2.)*(x[1]-2.) ) - 0.25 < 1e-6 ? 1.0 : 0.0", degree=2)
       return Expression("((x[0] == 2. & x[1] <= 2.45 & x[1] >=2.25) || (x[0] == 2. & x[1] <= 1.75 & x[1] >=1.55))? 1.0 : 0.0", degree=1)

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
        bc_6 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 6)
        bc_7 = DirichletBC(self.V_alpha, Constant(1.0), self.exterior_facets_meshfunction, 7)
        bc_8 = DirichletBC(self.V_alpha, Constant(1.0), self.exterior_facets_meshfunction, 8)
        return [bc_1, bc_2, bc_3, bc_4, bc_7,bc_8]

    def set_user_post_processing(self):


	output_file_u = HDF5File(mpi_comm_world(), "u_4_opening.h5", "w") # self.save_dir + "uO.h5"
	output_file_u.write(self.u, "solution")
	output_file_u.close()

	output_file_alpha = HDF5File(mpi_comm_world() , "alpha_4_opening.h5", "w")
	output_file_alpha.write(self.alpha, "solution")
	output_file_alpha.close()	



		
if __name__ == '__main__':

    # Run a fast simulation
    problem = Fracking(hsize=0.01, ell=1.0e-2, P_constant=0.1) #hsize=0.1, ell=1.0e-5, P_constant=1.)
    problem.solve()



