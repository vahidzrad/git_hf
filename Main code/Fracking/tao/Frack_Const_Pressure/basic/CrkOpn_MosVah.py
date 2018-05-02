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



def Opening(hsize,ell,law):
	# Geometry
	L = 4.0 # length
	H = 4.0 # height
	# Material constants
	#ell = Constant(4*hsize) # internal length scale
	#law = "AT1"

	prefix = "%s-L%s-H%.2f-S%.4f-l%.4f"%(law,L,H,hsize, ell)
	save_dir = "Fracking_result/" + prefix + "/"


	# I. We need to define mesh
	mesh = Mesh("meshes/fracking_hsize"+str(float(hsize))+".xml")
	#plot(mesh)

	# II. We need to define function spaces
	V_u = VectorFunctionSpace(mesh, "CG", 1)
	V_alpha = FunctionSpace(mesh, "CG", 1)


	# To read the solution for u_ and alpha_ (u,alpha)
	u = Function(V_u)
	input_file_u = HDF5File(mesh.mpi_comm(), save_dir+"u_4_opening.h5", "r")
	input_file_u.read(u, "solution")
	input_file_u.close()

	alpha = Function(V_alpha)
	input_file_alpha = HDF5File(mesh.mpi_comm(), save_dir+"alpha_4_opening.h5", "r")
	input_file_alpha.read(alpha, "solution")
	input_file_alpha.close()

	Volume = assemble( -inner(u,grad(alpha)) * dx)
	#print  "Volume=", Volume

	f_ud = -0.5*inner(u, grad(alpha))
	ud = project(f_ud, V_alpha)
	#########################################################################
	#This part is added by Mostafa (from last version of Vahid's code)
	#########################################################################
	# To draw a cubic spline and find out the closest point projection
	xp=np.array([1.8,1.9,2.0,2.1,2.2])
	yp=np.array([2.0,2.0,2.0,2.0,2.0])
	# To draw a spline
	cs=UnivariateSpline(xp,yp)
	xx=np.arange(1.8,2.2,0.01)
	yy = cs(xx)
	# We need to somehow automatically generate more DOF
	auto_dof_x = np.arange(1.8,2.2,0.01)  #Mostafa: What is this line? Is it the crack coordinate (s_direction)?
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

	bc_dof = [[1.8,2.],[2.2,2.]] #Mostafa: what is this line? the crack coordinates?
	bc_i = [map_vertex_to_dof[vx.index()] for j in range(len(bc_dof)) for vx in vertices(mesh_h) if np.allclose(mesh_h.coordinates()[vx.index()],bc_dof[j])]
	set_np = np.asarray(bc_i)

	lst_Coor_plt_X = []
	lst_open = []
	for vx in vertices(mesh_h):
	    lst_Coor_plt_X.append(mesh_h.coordinates()[vx.index()][0])

	arr_Coor_plt_X = np.asarray(lst_Coor_plt_X)


	#########################################################################
		#This part is added by Mostafa (from last version of Vahid's code)
	#########################################################################
	# To draw a cubic spline and find out the closest point projection
	xp=np.array([1.8,1.9,2.0,2.1,2.2])
	yp=np.array([2.0,2.0,2.0,2.0,2.0])
	# To draw a spline
	cs=UnivariateSpline(xp,yp)
	xx=np.arange(1.8,2.2,0.01)
	yy = cs(xx)
	# We need to somehow automatically generate more DOF
	auto_dof_x = np.arange(1.8,2.2,0.01)  #Mostafa: What is this line? Is it the crack coordinate (s_direction)?
	auto_dof_y = cs(auto_dof_x)

	editor = MeshEditor()
	mesh_h = Mesh()
		 
	editor.open(mesh_h,'interval',1,2)  #Mostafa: Are you mapping the 1D mesh(crack) on 2D domain?
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

	bc_dof = [[1.8,2.],[2.2,2.]] #Mostafa: what is this line? the crack coordinates?
	bc_i = [map_vertex_to_dof[vx.index()] for j in range(len(bc_dof)) for vx in vertices(mesh_h) if np.allclose(mesh_h.coordinates()[vx.index()],bc_dof[j])]
	set_np = np.asarray(bc_i)

	lst_Coor_plt_X = []
	lst_open = []
	for vx in vertices(mesh_h):
		lst_Coor_plt_X.append(mesh_h.coordinates()[vx.index()][0])

	arr_Coor_plt_X = np.asarray(lst_Coor_plt_X)

	#########################################################################

	#########################################################################
	# Added by Vahid for line integration (Wheeler's approach)
	# We should make a loop to compute this value for all nodes we need
	# Question: What are the nodes we are looking for? The nodes on mesh_h
	# Way II: To create a vector and ...
	# to do: We just need to write a 'for' loop to change li_idx based on nodes in mesh_h
	li_idx = np.arange(1.8, 2.201, 0.01)  # This is to calculate line integral for each DOF on mesh_h
	lst_li = []
	for i in range(len(li_idx)):
	    li_dof_y = np.arange(0.0, 4.01, 0.01)
	    li_dof_x = li_idx[i] * np.ones(len(li_dof_y))

	    editor = MeshEditor()
	    mesh_li = Mesh()

	    editor.open(mesh_li,'interval', 1, 2)
	    editor.init_vertices(len(li_dof_y))
	    editor.init_cells(len(li_dof_y) - 1)

	    for i in range(len(li_dof_y)):
		editor.add_vertex(i, np.array([round(li_dof_x[i], 2), round(li_dof_y[i], 2)]))

	    for j in range(len(li_dof_y) - 1):
		editor.add_cell(j, np.array([j, j + 1], dtype=np.uintp))

	    editor.close()

	    V_hl = FunctionSpace(mesh_li, "CG", 1)
	    f_hl = interpolate(ud, V_hl)
	    int_hl = assemble(f_hl * dx(mesh_li))
	    lst_li.append(int_hl)

	arr_li = np.asarray(lst_li)
	for idx in range(len(set_np)):
	    arr_li[set_np[idx]] = 0.



	return arr_Coor_plt_X, arr_li, Volume


if __name__ == '__main__':
  	 	 # test1.py executed as script
  	 	 # do something
  	 	 Opening()



