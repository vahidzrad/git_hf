# Code for outputting crack opening based on Bourdin's approach
# Author:
# Vahid Ziaei-Rad <vahidzrad@gmail.com>

import math
import os
import sys

import numpy as np
import sympy


from dolfin import *
from mshr import *

import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import LinearNDInterpolator

import petsc4py
petsc4py.init()
from petsc4py import PETSc
def vec(z):
    if isinstance(z, dolfin.cpp.Function):
        return dolfin.as_backend_type(z.vector()).vec()
    else:
       return dolfin.as_backend_type(z).vec()

def mat(A):
        return dolfin.as_backend_type(A).mat()

# Setup and parameters
set_log_level(INFO)



def BC_Stress(Mehtod, H, L, Lx):
	prefix = "L%s-H%.2f-%s"%(L,H,Mehtod)
	save_dir = "Elasticity_result/" + prefix + "/"


	# I. We need to define mesh
	mesh = RectangleMesh(Point(0., 0.), Point(L, H), Lx, Lx)
	#plot(mesh)

	# II. We need to define function spaces
	V_u = VectorFunctionSpace(mesh, "CG", 1)

	# To read the solution for u_ and alpha_ (u,alpha)
	u = Function(V_u)
	input_file_u = HDF5File(mesh.mpi_comm(), save_dir+"u_4_bc.h5", "r")
	input_file_u.read(u, "solution")
	input_file_u.close()


# To draw a cubic spline and find out the closest point projection
	xp=np.array([0.0,42.5,85,127.5,170])
	yp=np.array([170,170,170,170,170])

	# To draw a spline
	cs=UnivariateSpline(xp,yp)
	xx=np.arange(0 ,170, 1)
	yy = cs(xx)
	# We need to somehow automatically generate more DOF
	auto_dof_x = np.arange(0 ,170, 1)  #Mostafa: What is this line? Is it the crack coordinate (s_direction)?
	auto_dof_y = cs(auto_dof_x)

	editor = MeshEditor()
	mesh_h = Mesh()
	 
	editor.open(mesh_h,'interval', 1, 2)  #Mostafa: Are you mapping the 1D mesh(crack) on 2D domain?
	editor.init_vertices(len(auto_dof_x))
	editor.init_cells(len(auto_dof_x)-1)

	for i in range(len(auto_dof_x)):
	    editor.add_vertex(i, np.array([round(auto_dof_x[i],2), round(auto_dof_y[i],2)]))

	for j in range(len(auto_dof_x)-1):
	    editor.add_cell(j, np.array([j, j+1], dtype=np.uintp))

	editor.close()

	# One thing to fix: equal spaces on the spline, i.e., s-coordinate not x
	V_h = FunctionSpace(mesh_h, "CG", 1)

	# A simple way of imposing 0 for Vec_np
	map_vertex_to_dof = vertex_to_dof_map(V_h)
	map_dof_to_vertex = dof_to_vertex_map(V_h)

	bc_dof = [[0,170],[170,170]] #Mostafa: what is this line? the crack coordinates?
	bc_i = [map_vertex_to_dof[vx.index()] for j in range(len(bc_dof)) for vx in vertices(mesh_h) if np.allclose(mesh_h.coordinates()[vx.index()],bc_dof[j])]
	set_np = np.asarray(bc_i)

	lst_Coor_plt_X = []
	lst_open = []
	for vx in vertices(mesh_h):
	    lst_Coor_plt_X.append(mesh_h.coordinates()[vx.index()][0])

	arr_Coor_plt_X = np.asarray(lst_Coor_plt_X)


if __name__ == '__main__':
  	 	 # test1.py executed as script
  	 	 # do something
  	 	 BC_Stress()



