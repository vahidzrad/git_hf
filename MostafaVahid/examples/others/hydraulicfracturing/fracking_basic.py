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
        p.time.max = self.P_constant
        p.time.nsteps = 3 # Vahid: Just 3 steps for the whole pressure implementation? Isn't it too few?


        p.material.ell = self.ell
        #p.material.ell = 1e-1 #
        p.problem.hsize = self.hsize
        #p.material.Gc = 1/(1+self.hsize/(2*self.ell)) #AT1
        p.material.Gc = 1/(1+self.hsize/(2*self.ell)) #AT2 # Vahid: Don't understand: Why 'Gc' is like that?

        #p.material.Gc = 1.5e6
        p.material.E = 1.0
        p.material.nu = 0.0 # Vahid: Why 'nu' is zero? If the material is incompressible, shouldn't it be selected '0.5'?
        p.material.law = "AT2" # Vahid: 'AT2' is much easier to solve, but harder to converge! Can't we also try with 'AT1'?
        p.post_processing.plot_alpha = False
        p.post_processing.plot_u = False
        p.post_processing.save_alpha = False
        p.post_processing.save_u = False
        p.post_processing.save_Beta = False
        p.post_processing.save_V = False
        p.solver_alpha.method = "gpcg" # Vahid: We should also seek for other solving methods
        p.post_processing.save_energies = True

		#lc = DefineNumber[ %g, Name "Parameters/lc" ]; 
    def define_mesh(self): # Vahid: So, here we just recall the mesh from 'Diffusion_gas.py'?
        geofile = \
		"""
		
		lc = DefineNumber[ %g, Name "Parameters/lc" ];
                Point(1) = {0, 0, 0, 1*lc};
                Point(2) = {4, 0, 0, 1*lc};
                Point(3) = {4, 4, 0, 1*lc};
                Point(4) = {0, 4, 0, 1*lc};
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

		Physical Line(101) = {6}; # Vahid: You can use a simpler label other than '101'. I think strings like 'l' would also work.

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
                return near(x[1], 2) and  x[0] <= 2.2 and x[0] >= 1.8 and on_boundary # Vahid: a more beautiful way: abs(x_0 - 2) < .2 The same applies for the rest three intervals

	class Void(SubDomain): # Vahid: So, is Void the place where the pressure is applied? Why not on the whole boundary instead?
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

   # def P_b(self):
	#P_b= Constant(0.1)
	#P_b= Expression("t", t=0.0, degree =1)
    	#return P_b

    def define_pressure(self):
	pressure = Function(self.V_alpha)
	input_file_pressure = HDF5File(self.mesh.mpi_comm(), "pressure.h5", "r") # Vahid: Have you checked this file? It MIGHT be a damaged file!
	input_file_pressure.read(pressure, "solution")
	input_file_pressure.close()

	#pressure_n=100000.*pressure.vector().array()

	#output_file_pressure = HDF5File(mpi_comm_world() , "pressure_new.h5", "w")
	#output_file_pressure.write(pressure_n, "solution")
	#output_file_pressure.close()
    	return pressure


    def define_initial_alpha(self):
       return Expression("x[1] == 2. & x[0] <= 2.2 & x[0] >=1.8 ? 1.0 : 0.0", degree=1)
	# Vahid: Isn't it better to use 'near(x[1], 2)' instead? If this function is applied on nodes, there is no guarantee that there are nodes with 'x[1]=2'
       #return Expression("((x[0] == 2. & x[1] <= 2.45 & x[1] >=2.25) || (x[0] == 2. & x[1] <= 1.75 & x[1] >=1.55))? 1.0 : 0.0", degree=1)

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
        #bc_7 = DirichletBC(self.V_alpha, Constant(1.0), self.exterior_facets_meshfunction, 7)
        #bc_8 = DirichletBC(self.V_alpha, Constant(1.0), self.exterior_facets_meshfunction, 8)
        return [bc_1, bc_2, bc_3, bc_4, bc_5]

    def set_user_post_processing(self):
	output_file_u = HDF5File(mpi_comm_world(), "u_4_opening.h5", "w") # self.save_dir + "uO.h5"
	output_file_u.write(self.u, "solution")
	output_file_u.close()

	output_file_alpha = HDF5File(mpi_comm_world() , "alpha_4_opening.h5", "w")
	output_file_alpha.write(self.alpha, "solution")
	output_file_alpha.close()	



		
if __name__ == '__main__':

    # Run a fast simulation
    problem = Fracking(hsize=0.1, ell=4*1.0e-2, P_constant=0.5) #hsize=0.1, ell=1.0e-5, P_constant=1.)
	# Vahid: Do you really use these parameters? If this is the case, then why 'h' is too big? h/ell=2.5, normally h/ell<.5
    problem.solve()



