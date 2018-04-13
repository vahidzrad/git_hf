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

from fenics import *

def default_parameters():

    p = Parameters("user_parameters")
    subset_list = ["problem", "material", "fem", "time", "AM", "solver_alpha", "solver_u", "solver_slepc", "post_processing"]
    for subparset in subset_list:
        subparset_is = eval("default_" + subparset + "_parameters()")
        p.add(subparset_is)
    return p

def default_problem_parameters():

    problem = Parameters("problem")
    problem.add("hsize", 1e-3) # Vahid: So, is 'hsize' defined here, and also redefine in 'fracking_basic.py'? The same question applies for 'E', 'nu', 'Gc', 'P_constant', and perhaps etc.
    problem.add("stability", False)
    problem.add("stability_correction", False)

    return problem

def default_material_parameters():

    material = Parameters("material")
    material.add("E", 1.0)
    material.add("nu", 0.0)
    material.add("kres", 1e-6)
    material.add("Gc", 1.0)
    material.add("ell", 1e-2)
    material.add("law", "AT1") # Vahid: Here law is 'AT1', but in 'fracking_basic.py', it is 'AT2'
    material.add("k", 2.0)
    material.add("pstress", False)
    material.add("C_biot", 0)
    material.add("P_constant", 0.0)

    return material

def default_fem_parameters():

    fem = Parameters("fem")
    fem.add("u_degree", 1)
    fem.add("alpha_degree", 1)

    return fem

def default_time_parameters():

    time = Parameters("time")
    time.add("min", 0.0)
    time.add("max", 1.0)
    time.add("nsteps", 50)

    return time

def default_AM_parameters():

    AM = Parameters("AM")
    AM.add("max_iterations", 100)
    AM.add("tolerance", 1e-5)

    return AM

def default_solver_u_parameters():

    solver_u = Parameters("solver_u")
    solver_u.add("linear_solver", "mumps") # prefer "superlu_dist" or "mumps" if available
    solver_u.add("preconditioner", "default")
    solver_u.add("report", False)
    solver_u.add("maximum_iterations", 500) #Added by Mostafa # Vahid: Why? And how did you determine these numbers?
    solver_u.add("relative_tolerance", 1e-6) #Added by Mostafa

    return solver_u

def default_solver_alpha_parameters():

    solver_alpha = Parameters("solver_alpha")
    solver_alpha.add("method", "tron")
    solver_alpha.add("linear_solver", "stcg")
    solver_alpha.add("preconditioner", "jacobi")
    solver_alpha.add("line_search", "gpcg")

    return solver_alpha

def default_solver_slepc_parameters():

    solver_slepc = Parameters("solver_slepc")
    solver_slepc.add("method", "lobpcg")
    solver_slepc.add("linear_solver", "cg")
    solver_slepc.add("preconditioner", "hypre_amg")
    solver_slepc.add("tolerance", 1e-6)
    solver_slepc.add("report", False)
    solver_slepc.add("monitor", False)
    solver_slepc.add("maximum_iterations", 0)

    return solver_slepc

def default_post_processing_parameters():

    post_processing = Parameters("post_processing")
    post_processing.add("save_energies", False)
    post_processing.add("save_u", False)
    post_processing.add("save_alpha", False)
    post_processing.add("save_V", False)
    post_processing.add("save_Beta", False)
    post_processing.add("plot_u", False)
    post_processing.add("plot_alpha", False)
    post_processing.add("file_u", "u.xdmf")
    post_processing.add("file_alpha", "alpha.pvd")
    post_processing.add("file_V", "V.xdmf")
    post_processing.add("file_Beta", "Beta.xdmf")
    post_processing.add("file_energies", "energies.txt")

    return post_processing
