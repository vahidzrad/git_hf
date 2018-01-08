# Copyright (C) 2015 Corrado Maurini, Tianyi Li
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

# Problem
class NucleationBar(QuasiStaticGradentDamageProblem):

    def __init__(self, ell, hsize=0.002, W=0.1, law="AT1", k=1.0):
        self.ell = ell
        self.hsize = hsize
        self.W = W
        self.law = law
        self.k = k
        QuasiStaticGradentDamageProblem.__init__(self)
        self.F = []

    def prefix(self):
        return "bar"

    def set_user_parameters(self):
        p = self.parameters

        p.time.min = 0.0
        p.time.max = 5.0
        p.time.nsteps = 500

        p.problem.hsize = self.hsize

        p.material.ell = self.ell
        p.material.law = self.law
        p.material.k = self.k
        if self.law == "AT1":
            p.material.Gc = 8*self.ell/3
        elif self.law == "AT2":
            p.material.Gc = 3*self.ell
        elif self.law == "ATk":
            p.material.Gc = pl.pi*self.k*self.ell/2

        p.AM.max_iterations = 1000

        p.post_processing.save_energies = False
        p.post_processing.save_u = False
        p.post_processing.save_alpha = True

        if p.material.law != "ATk":
            p.solver_alpha.method = "gpcg"
            p.solver_alpha.linear_solver = "nash"

    def set_time_stepping(self):
        time = self.parameters.time
        if self.parameters.material.law != "AT2":
            phase1 = pl.array([0, 1])
        else:
            phase1 = pl.linspace(0, 1, 50)
        eps = 1e-4
        phase2 = pl.linspace(1+eps, time.max, time.nsteps)
        self.time_steps = pl.concatenate([phase1, phase2])
        self.user_break = False

    def define_mesh(self):
        geofile = \
            """
            hsize = DefineNumber[ %g, Name "Parameters/hsize" ];
            W = DefineNumber[ %g, Name "Parameters/W" ];
            Point(1) = {0, 0, 0, hsize};
            Point(2) = {1, 0, 0, hsize};
            Point(3) = {1, W, 0, hsize};
            Point(4) = {0, W, 0, hsize};
            Line(1) = {1, 2};
            Line(2) = {2, 3};
            Line(3) = {3, 4};
            Line(4) = {4, 1};
            Line Loop(1) = {1, 2, 3, 4};
            Plane Surface(1) = {1};
            Physical Surface(0) = {1};
            """ % (self.hsize, self.W)

        return mesher(geofile, self.prefix())

    def set_user_post_processing(self):
        alpha_max = self.alpha.vector().max()
        e1 = Constant([1., 0.])
        F = assemble(inner(self.materials[0].sigma(self.u, self.alpha)*e1, e1)*self.ds(2))/self.W  # normalized stress

        if alpha_max < 0.99:
            U = self.t
        else:
            U = (self.time_steps[self.step-1] + self.time_steps[self.step])/2
            self.user_break = True

        self.F.append([U, F])
        pl.savetxt(self.save_dir + "F.txt", pl.array(self.F), "%.5e")

    def set_mesh_functions(self):
        self.cells_meshfunction = MeshFunction("size_t", self.mesh, self.dimension)
        self.cells_meshfunction.set_all(0)
        self.exterior_facets_meshfunction = MeshFunction("size_t", self.mesh, self.dimension - 1)
        self.exterior_facets_meshfunction.set_all(0)

        class Left(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0.)

        class Right(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 1.)

        Left().mark(self.exterior_facets_meshfunction, 1)
        Right().mark(self.exterior_facets_meshfunction, 2)

    def define_bc_u(self):
        bc_1 = DirichletBC(self.V_u.sub(0), Constant(0.0), self.exterior_facets_meshfunction, 1)
        bc_2 = DirichletBC(self.V_u.sub(0), Expression("t", t=0.0, degree = 0), self.exterior_facets_meshfunction, 2)
        return [bc_1, bc_2]

    def define_bc_alpha(self):
        bc_1 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 1)
        bc_2 = DirichletBC(self.V_alpha, Constant(0.0), self.exterior_facets_meshfunction, 2)
        return [bc_1, bc_2]

if __name__ == '__main__':

    # Run a fast simulation
    problem = NucleationBar(ell=0.01, hsize=0.01, W=0.1, law="AT1")
    problem.solve()
