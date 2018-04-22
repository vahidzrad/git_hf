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

from __future__ import division
from petsc4py import PETSc
from fenics import *
from default_parameters import *
from material import *
from utils import *
import pylab as pl
import hashlib
import json
import os

class QuasiStaticGradentDamageProblem(object):

    """
    Problem class for quasi-static gradient damage models
    """

    ## Constructor
    def __init__(self):

        # Set the mpi communicator of the object
        self.comm_rank = MPI.rank(mpi_comm_world())
        self.comm_size = MPI.size(mpi_comm_world())

        # Parameters
        self.parameters = default_parameters()
        self.set_user_parameters()
        self.parameters.parse()

        # Mesh
        self.mesh = self.define_mesh()
        self.dimension = self.mesh.geometry().dim()

        # MeshFunctions and Measures for different blocks and boundaries
        self.set_mesh_functions()
        self.set_measures()

        # Materials
        self.materials = self.define_materials()

        # Variational formulation
        self.set_variational_formulation()

        # Dirichlet BC
        self.bc_u = self.define_bc_u()
        self.bc_alpha = self.define_bc_alpha()


        # Time-stepping
        self.set_time_stepping()

        # Post-processing
        self.save_dir = self.generate_save_dir()
        pp = self.parameters.post_processing
        #self.file_u = XDMFFile(mpi_comm_world(), self.save_dir + pp.file_u)
        #self.file_alpha = XDMFFile(mpi_comm_world(), self.save_dir + pp.file_alpha)

	self.file_alpha = File(self.save_dir +"/alpha.pvd") # use .pvd if .xdmf is not working
	self.file_u = File(self.save_dir +"/u.pvd") # use .pvd if .xdmf is not working

        self.file_V = XDMFFile(mpi_comm_world(), self.save_dir + pp.file_V)
        self.file_Beta = XDMFFile(mpi_comm_world(), self.save_dir + pp.file_Beta)
        self.print_parameters()
        self.energies = []

    def print0(self, text):
        """
        Print only from process 0 (for parallel run)
        """
        if self.comm_rank == 0: print(text)

    def prefix(self):
        return "problem"

    def generate_save_dir(self):
        data = self.parameters.to_dict()
        signature = hashlib.md5(json.dumps(data, sort_keys=True)).hexdigest()
        savedir = self.prefix() + "-" + "results" + "/" + signature + "/"
        self.create_save_dir(savedir)
        return savedir

    def create_save_dir(self,savedir):
        if self.comm_rank == 0:
            if os.path.isdir(savedir) == False:
                if os.path.isdir(savedir.split("/")[0]) == False:
                    os.mkdir(savedir.split("/")[0])
                os.mkdir(savedir)
        MPI.barrier(mpi_comm_world())

    def print_parameters(self):
        if self.comm_rank == 0:
            file_parameters = File(self.save_dir + "parameters.xml")
            file_parameters << self.parameters
            import json
            with open(self.save_dir + 'parameters.pkl', 'w') as f:
                json.dump(self.parameters.to_dict(), f)

    def define_mesh(self):
        pass

    def set_measures(self):
        """
        Here we assign the Measure to get selective integration on boundaries and bulk subdomain
        The Measure are defined using self.cells_meshfunction and self.exterior_facets_meshfunction
        """
        try:
            self.dx = Measure("dx")(subdomain_data=self.cells_meshfunction)
        except Exception:
            self.dx = dx
        try:
            self.ds = Measure("ds")(subdomain_data=self.exterior_facets_meshfunction)
        except Exception:
            self.ds = ds

    def set_mesh_functions(self):
        """
        Set here meshfunctions with boundaries and subdomains indicators
        """
        self.cells_meshfunction = MeshFunction("size_t", self.mesh, self.dimension)
        self.cells_meshfunction.set_all(0)
        self.exterior_facets_meshfunction = MeshFunction("size_t", BoundaryMesh(self.mesh, "exterior"), self.dimension - 1)
        self.exterior_facets_meshfunction.set_all(0)

    def define_bc_u(self):
        """
        Return a list of boundary conditions on the displacement
        """
        return []

    def define_bc_alpha(self):
        """
        Return a list of boundary conditions on the damage field
        """
        return []




    def define_materials(self):
        """
        Return list of materials that will be set in the model.
        """
        material1 = GradientDamageMaterial(self.parameters.material, "everywhere")
        return [material1]

    def set_user_parameters(self):
        """
        Set with this function the user parameters defining the problem
        """
        pass

    def set_time_stepping(self):
        """
        Set here the discretized time steps in self.time_steps
        """
        time = self.parameters.time
        self.time_steps = pl.linspace(time.min, time.max, time.nsteps)
        self.user_break = False

    def set_loading(self, t):
        """
        Update Dirichlet boundary conditions and inelastic strains
        """
        for bcs in [self.bc_u, self.bc_alpha]:
            for bc in bcs:
                if hasattr(bc.function_arg, "t"):
                    bc.function_arg.t = t

        for material in self.materials:
            if hasattr(material._eps0, "t"):
                material._eps0.t = t

    def define_user_energy(self):
        """
        Add here an arbitrary term in the energy functional
        (for example the potential energy of surface forces)
        """
        return 0

    def define_initial_alpha(self):
        """
        Return the initial damage state.
        Admissabile damage fields have damage levels not smaller than this state
        """
        return Constant(0.0)

    def define_initial_guess_alpha(self):
        """
        Return the initial guess for the damage.
        Admissabile damage fields can have damage levels smaller than this state
        """
        return Constant(0.0)

    def set_user_post_processing(self):
        """
        User post-processing (to break time iteration for example when crack has been created)
        """
        pass

    def set_user_job(self):
        """
        User job inside the alternate minimization
        """
        pass

    def set_variational_formulation(self):
        """
        Define the variational problem to be solved
        """
        fem = self.parameters.fem

        # Create function spaces
        element_u = VectorElement("CG", self.mesh.ufl_cell(), fem.u_degree)
        element_alpha = FiniteElement("CG", self.mesh.ufl_cell(), fem.alpha_degree)
        element_u_alpha = MixedElement([element_u, element_alpha])
        self.V_u = FunctionSpace(self.mesh, element_u)
        self.V_alpha = FunctionSpace(self.mesh, element_alpha)
        self._V_u_alpha = FunctionSpace(self.mesh, element_u_alpha)

        # Solution, test and trial functions
        self.u = Function(self.V_u, name="Displacement")
        self.du = TrialFunction(self.V_u)
        self.v = TestFunction(self.V_u)

        self.alpha = Function(self.V_alpha, name="Damage")
        self.alpha_prev = Function(self.V_alpha)
        self.alpha_ub = Function(self.V_alpha)
        self.dalpha = TrialFunction(self.V_alpha)
        self.beta = TestFunction(self.V_alpha)



        # Initialize alpha
        self.alpha.interpolate(self.define_initial_guess_alpha())
        self.alpha_prev.interpolate(self.define_initial_alpha())
        self.alpha_ub.interpolate(Constant(1.0))

        # Energies
        self.elastic_energy = sum([material.elastic_energy_density(self.u, self.alpha)*self.dx(material.subdomain_id) for material in self.materials])
        self.dissipated_energy = sum([material.dissipated_energy_density(self.u, self.alpha)*self.dx(material.subdomain_id) for material in self.materials])
        self.total_energy = self.elastic_energy + self.dissipated_energy

        # First derivatives of energies
        self.Du_total_energy = derivative(self.total_energy, self.u, self.v)
        self.Dalpha_total_energy = derivative(self.total_energy, self.alpha, self.beta)

        # Second derivatives of energies
        self.J_u = derivative(self.Du_total_energy, self.u, self.du)
        self.J_alpha = derivative(self.Dalpha_total_energy, self.alpha, self.dalpha)



    def set_eigensolver_parameters(self):
        """
        Use slepc4py and petsc4py to set the user parameters
        """


    def solve_u(self):
        """
        Solve the displacement problem at fixed damage
        """
        problem_u = NonlinearVariationalProblem(self.Du_total_energy, self.u, self.bc_u, self.J_u)
        solver_u = NonlinearVariationalSolver(problem_u)
        solver_u.parameters.symmetric = True
        solver_u.parameters.nonlinear_solver = "newton"
        solver_u.parameters.newton_solver.update(self.parameters.solver_u)

        solver_u.solve()

    def solve_alpha(self):
        """
        Solve the damage problem at fixed displacement.
        """
        class DamageProblem(OptimisationProblem):

            def __init__(self, problem):
                OptimisationProblem.__init__(self)
                self.total_energy = problem.total_energy
                self.Dalpha_total_energy = problem.Dalpha_total_energy
                self.J_alpha = problem.J_alpha
                self.alpha = problem.alpha
                self.bc_alpha = problem.bc_alpha

            def f(self, x):
                self.alpha.vector()[:] = x
                return assemble(self.total_energy)

            def F(self, b, x):
                self.alpha.vector()[:] = x
                assemble(self.Dalpha_total_energy, b)
                for bc in self.bc_alpha:
                    bc.apply(b)

            def J(self, A, x):
                self.alpha.vector()[:] = x
                assemble(self.J_alpha, A)
                for bc in self.bc_alpha:
                    bc.apply(A)

        solver_alpha = PETScTAOSolver()
        solver_alpha.parameters.update(self.parameters.solver_alpha)
        solver_alpha.solve(DamageProblem(self), self.alpha.vector(), self.alpha_prev.vector(), self.alpha_ub.vector())

    def solve_step(self):
        """
        Solve the u-alpha problem at a given loading with the alternate-minimization algorithm
        """

        # Alternate minimization
        iteration = 0
        error = 1.0
        AM = self.parameters.AM
        alpha_old = self.alpha.copy(True)
        while iteration < AM.max_iterations and error > AM.tolerance and not self.user_break:
            self.solve_u()
            self.solve_alpha()
            alpha_diff = self.alpha.vector() - alpha_old.vector()
            error = alpha_diff.norm("linf")
            iteration += 1
            alpha_old.assign(self.alpha)
            self.print0("AM: Iteration number: %i - Error: %.3e" % (iteration, error))
            self.set_user_job()

    def solve_stability(self):
        """
        Solve the 2nd-order stability problem
        """



    def solve(self):
        """
        Solve the quasi-static evolution problem (time-stepping)
        This is the function to call to solve the problem
        """
        # Time-stepping
        for i, t in enumerate(self.time_steps):

            # Updating time
            self.t = float(t)
            self.step = i
            self.print0("\033[1;32m--- Time step %d: t = %g ---\033[1;m" % (i, t))
            self.set_loading(t)

            # Solving

            # Without 2nd-order stability analysis
            self.solve_step()

            # Damage irreversibility
            self.alpha_prev.assign(self.alpha)

            # Post-processing
            self.post_processing()

            # Allow user break
            if self.user_break:
                break

        # We are at the end of the calculation
        self.finalize()

    def post_processing(self):
        """
        Post-processing at the end of each time iteration
        """
        pp = self.parameters.post_processing

        # Plot or save fields
        if pp.plot_u and self.comm_size == 1:
            plot(self.u, mode = "displacement")

        if pp.plot_alpha and self.comm_size == 1:
            plot(self.alpha, mode = "color")

        if pp.save_u:
            #self.file_u.write(self.u, self.t)
	    self.file_u << (self.u, self.t)

        if pp.save_alpha:
            #self.file_alpha.write(self.alpha, self.t)
	    self.file_alpha << (self.alpha, self.t)

        if pp.save_V:
            self.file_V.write(self.V, self.t)

        if pp.save_Beta:
            self.file_Beta.write(self.Beta, self.t)

        # Calculate energies
        if pp.save_energies:
            self.elastic_energy_value = assemble(self.elastic_energy)
            self.dissipated_energy_value = assemble(self.dissipated_energy)
            if self.comm_rank == 0:
                self.energies.append([self.t, self.elastic_energy_value, self.dissipated_energy_value])
                pl.savetxt(self.save_dir + pp.file_energies, pl.array(self.energies), "%.5e")

        # User post-processing
        self.set_user_post_processing()

    def finalize(self):
        """
        This is executed at the end of the calculation
        """
        # Plot energies with matplotlib
        pp = self.parameters.post_processing
        if pp.save_energies and self.comm_rank == 0:
            self.energies = pl.array(self.energies)
            pl.plot(self.energies[:, 0], self.energies[:, 1], label = "Elastic")
            pl.plot(self.energies[:, 0], self.energies[:, 2], label = "Dissipated")
            pl.plot(self.energies[:, 0], self.energies[:, 1] +  self.energies[:, 2], label = "Total")
            pl.xlabel("Time")
            pl.ylabel("Energies")
            pl.legend(loc = "best")
            pl.savefig(self.save_dir + pp.file_energies.replace("txt", "pdf"), transparent=True)
            pl.close()
