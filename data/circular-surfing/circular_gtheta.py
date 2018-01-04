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

import matplotlib
from fenics import *
import pylab as pl
import h5py

mesh = Mesh("meshes/circular.xdmf")
V = VectorFunctionSpace(mesh, "CG", 1)
u = Function(V)
V_alpha = FunctionSpace(mesh, "CG", 1)
alpha = Function(V_alpha)
dim = len(u)

Q = FunctionSpace(mesh, "CG", 1)
q = Function(Q)
v2d = vertex_to_dof_map(Q)

study_case = "circular-results/738b2df574f49d3215bedba48daf06f4"
fl_u = h5py.File(study_case + "/u.h5", "r")
fl_alpha = h5py.File(study_case + "/alpha.h5", "r")

data = pl.loadtxt(study_case + "/energies.txt")
t, E, S = data[:, 0], data[:, 1], data[:, 2]

Gc = 1
h = 0.004
ell = 0.02
Gc_num = (1+3*h/(8*ell))*Gc

f_u = File("u.xdmf")
f_alpha = File("alpha.xdmf")
f_theta = File("theta.xdmf")
theta = Function(V, name="Virtual perturbation")

# Varying the inner radius
G = pl.zeros(len(t))

for i in range(len(t)):

    # Obtain the displacement field
    vec = fl_u["/VisualisationVector/" + str(i)]
    for j in range(2):
        q.vector()[v2d] = vec[:, j]
        assign(u.sub(j), q)
    # f_u << u

    # Obtain the damage field
    vec = fl_alpha["/VisualisationVector/" + str(i)]
    alpha.vector()[v2d] = vec[:, 0]
    # f_alpha << alpha

    # Construct the theta field
    x0 = cos(t[i])
    y0 = sin(t[i])
    r0 = 4*ell
    r1 = 10*ell

    def neartip(x, on_boundary):
        dist = sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
        return dist < r0
    def outside(x, on_boundary):
        dist = sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
        return dist > r1
    bc1 = DirichletBC(V, Constant([1.0, 1.0]), neartip)
    bc2 = DirichletBC(V, Constant([0.0, 0.0]), outside)
    bc = [bc1, bc2]
    scaling = Function(V)
    theta_trial = TrialFunction(V)
    theta_test = TestFunction(V)
    a = inner(grad(theta_trial), grad(theta_test))*dx
    L = inner(Constant([0.0, 0.0]), theta_test)*dx
    solve(a == L, scaling, bc, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
    theta.interpolate(Expression(["-x[1]/sqrt(x[0]*x[0]+x[1]*x[1])", "+x[0]/sqrt(x[0]*x[0]+x[1]*x[1])"]))
    theta.vector()[:] = scaling.vector()*theta.vector()
    f_theta << theta, float(t[i])

    # G-theta method
    Id = Identity(dim)
    E = 1.0
    nu = 0.0
    mu = E/(2.0*(1.0+nu))
    lmbda = E*nu/(1.0-nu**2)
    eps = sym(grad(u))
    sigma0 = Constant(2.0)*Constant(mu)*eps+Constant(lmbda)*tr(eps)*Id
    sigma = (1-alpha)**2*sigma0
    Gtheta = assemble(inner(sigma, dot(grad(u), grad(theta)))*dx - Constant(0.5)*inner(sigma, eps)*div(theta)*dx)/Gc_num

    G[i] = Gtheta
    print("Gtheta = %.3e" %(Gtheta))
    pl.savetxt("Gtheta.txt", pl.array(G), "%.5e")

pl.plot(t, G)
pl.xlabel("Prescribed angle")
pl.ylabel(r"Energy release rate $G^\alpha/(G_\mathrm{c})_\mathrm{eff}$")
pl.savefig("gtheta_circular.pdf")