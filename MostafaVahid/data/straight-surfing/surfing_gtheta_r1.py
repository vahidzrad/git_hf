# Copyright (C) 2014 Tianyi Li
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

from edfpylab import *
from fenics import *
import pylab as pl
import h5py

mesh = Mesh("meshes/surfing_h0.005.xml")
V = VectorFunctionSpace(mesh, "CG", 1)
u = Function(V)
V_alpha = FunctionSpace(mesh, "CG", 1)
alpha = Function(V_alpha)
dim = u.domain().geometric_dimension()

Q = FunctionSpace(mesh, "CG", 1)
q = Function(Q)
v2d = vertex_to_dof_map(Q)

study_case = "surfing-results/bbf4a1b513046497aa531031c1d5a2fb"
#            "surfing-results/e92d9ba017500101dad2f0b8aca22d78"  # ATk with k = 10

fl_u = h5py.File(study_case + "/u.h5", "r")
fl_alpha = h5py.File(study_case + "/alpha.h5", "r")

data = pl.loadtxt(study_case + "/energies.txt")
t, E, S = data[:, 0], data[:, 1], data[:, 2]

Gc = 1.5
h = 0.005
ell = 0.025
Gc_eff_AT1 = (1+3*h/(8*ell))*Gc
Gc_eff_ATk = (1+h/(pl.pi*ell))*Gc

f_alpha = File("alpha.xdmf")
f_theta = File("theta.xdmf")

# Varying the inner radius
r_list = pl.linspace(1, 5, 10)*ell
t_list = [50, 60, 70]
G = pl.zeros([len(r_list), len(t_list)])
Gc_value = pl.zeros([len(r_list), len(t_list)])

for i, t in enumerate(t_list):
    for k, r in enumerate(r_list):

        # Obtain the displacement field
        vec = fl_u["/VisualisationVector/" + str(t)]
        for j in range(2):
            q.vector()[v2d] = vec[:, j]
            assign(u.sub(j), q)

        # Obtain the damage field
        vec = fl_alpha["/VisualisationVector/" + str(t)]
        alpha.vector()[v2d] = vec[:, 0]
        # f_alpha << alpha

        # Construct the theta field
        x0 = S[t]/Gc_eff_ATk
        r1 = 2.5*r

        def neartip(x, on_boundary):
            dist = sqrt((x[0]-x0)**2 + x[1]**2)
            return dist < r
        def outside(x, on_boundary):
            dist = sqrt((x[0]-x0)**2 + x[1]**2)
            return dist > r1
        bc1 = DirichletBC(V, Constant([1.0, 0.0]), neartip)
        bc2 = DirichletBC(V, Constant([0.0, 0.0]), outside)
        bc = [bc1, bc2]
        theta = Function(V)
        theta_trial = TrialFunction(V)
        theta_test = TestFunction(V)
        a = inner(grad(theta_trial), grad(theta_test))*dx
        L = inner(Constant([0.0, 0.0]), theta_test)*dx
        solve(a == L, theta, bc)
        # f_theta << theta

        # G-theta method
        Id = Identity(dim)
        E = 1.0
        nu = 0.2
        mu = E/(2.0*(1.0+nu))
        lmbda = E*nu/(1.0-nu**2)
        eps = sym(grad(u))
        sigma0 = Constant(2.0)*mu*eps+lmbda*tr(eps)*Id
        sigma = (1-alpha)**2*sigma0
        # sigma = (1-alpha)**2/(1+9*(1-(1-alpha)**2))*sigma0
        Gtheta = assemble(inner(sigma, grad(u)*grad(theta))*dx - Constant(0.5)*inner(sigma, eps)*div(theta)*dx)/Gc_eff_AT1

        G[k, i] = Gtheta
        print("gtheta = %.3e" %(Gtheta))
        pl.savetxt("gtheta.txt", pl.array(G), "%.5e")

pl.figure(1)
pl.plot(pl.linspace(1, 5, 10), G[:, 0], "bo-")
pl.plot(pl.linspace(1, 5, 10), G[:, 1], "gv-")
pl.plot(pl.linspace(1, 5, 10), G[:, 2], "r>-")
pl.xlabel("Inner radius $r/\ell$")
pl.ylabel(r"Energy release rate $G/(G_\mathrm{c})_\mathrm{eff}$")
pl.savefig("gtheta_of_r.ipe", format="ipe")
