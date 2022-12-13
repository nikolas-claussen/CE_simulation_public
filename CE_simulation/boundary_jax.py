# AUTOGENERATED! DO NOT EDIT! File to edit: ../04b_boundary_conditions_jax.ipynb.

# %% auto 0
__all__ = ['get_E_jac', 'get_E_dual_jac', 'get_triangular_lattice', 'create_rect_mesh', 'create_rect_mesh_angle', 'polygon_area',
           'polygon_perimeter', 'get_E', 'get_E_dual', 'get_conformal_transform', 'excitable_dt_act_pass',
           'get_flip_edge']

# %% ../04b_boundary_conditions_jax.ipynb 3
from .triangle import *
from .tension import *
from .delaunay import *
from .isogonal import *

# %% ../04b_boundary_conditions_jax.ipynb 4
import os
import sys
import importlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

from numpy import sin, cos, tan, pi, sqrt, arccos, arctan, arctan2
from numpy.linalg import norm

from scipy.integrate import solve_ivp
from scipy import ndimage
from scipy import spatial
from scipy import optimize
from scipy import linalg

from tqdm.notebook import tqdm

from copy import deepcopy

from collections import Counter, defaultdict

# %% ../04b_boundary_conditions_jax.ipynb 5
from dataclasses import dataclass
from typing import Union, Dict, List, Tuple, Iterable, Callable
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../04b_boundary_conditions_jax.ipynb 6
import ipywidgets as widgets
from matplotlib import animation, rc

# %% ../04b_boundary_conditions_jax.ipynb 7
import jax.numpy as jnp
from jax import grad as jgrad
from jax import jit
from jax.tree_util import Partial
from jax.config import config
from jax.nn import relu as jrelu

config.update("jax_enable_x64", True) # 32 bit leads the optimizer to complain about precision loss
#config.update("jax_debug_nans", True)  # useful for debugging, but makes code slower!

# %% ../04b_boundary_conditions_jax.ipynb 10
@patch
def get_centroid(self: Vertex):
    """Get centroid of dual cell"""
    return np.mean([x.dual_coords for x in self.get_face_neighbors() if x is not None], axis=0)

# %% ../04b_boundary_conditions_jax.ipynb 14
def get_triangular_lattice(nx, ny):
    """get triangular lattice with nx, ny points. Return a mask which delinates bdry vertices""" 

    y = np.arange(0, ny)*sqrt(3)/2
    x = np.arange(nx).astype(float)
    X, Y = np.meshgrid(x, y)
    X -= X.mean()+1/2; Y -=Y.mean()
    X = (X.T+(np.arange(ny)%2)/2).T
    pts = np.stack([X, Y]).reshape((2,nx*ny))
    is_bdry = np.zeros_like(X)
    is_bdry[:1] = is_bdry[-1:] = 1
    is_bdry[:,:1] = is_bdry[:,-1:] = 1
    is_bdry = is_bdry.reshape(nx*ny)
    
    return pts, is_bdry

def create_rect_mesh(nx, ny, noise=0, defects=(0,0), straight_bdry=False):
    pts, is_bdry = get_triangular_lattice(nx, ny)
    pts[:,~is_bdry.astype(bool)] += np.random.normal(scale=noise, size=(2, (~is_bdry.astype(bool)).sum()))
    if defects[0] > 0:
        ix = np.random.choice(np.where(1-is_bdry)[0], size=defects[0], replace=False)
        pts = np.delete(pts, ix, axis=1)
    if defects[1] > 0:
        ix = np.random.choice(np.where(1-is_bdry)[0], size=defects[1], replace=False)
        split = np.random.choice((0,1), len(ix))
        additional_pts =  pts[:, ix] + .3*np.stack([1-split, split]) 
        pts[:, ix] -= .3*np.stack([1-split, split]) 
        pts = np.hstack([pts, additional_pts])
    
    tri = spatial.Delaunay(pts.T)
    # remove the left, right edge
    if straight_bdry:
        simplices = tri.simplices
    else:
        max_x, min_x = (pts[0].max(), pts[0].min())
        simplices = np.stack([x for x in tri.simplices
                          if (np.isclose(pts[0,x], min_x).sum()<2) and (np.isclose(pts[0,x], max_x).sum()<2)])
    pre_mesh = ListOfVerticesAndFaces(tri.points, simplices)
    mesh = HalfEdgeMesh(pre_mesh)
    
    return mesh

# %% ../04b_boundary_conditions_jax.ipynb 15
def create_rect_mesh_angle(nx, ny, angle=0, noise=0, max_l=2):
    pts, _ = get_triangular_lattice(2*max([nx, ny]), 2*max([nx, ny]))
    pts = (pts.T - pts.mean(axis=1)).T
    pts = rot_mat(angle) @ pts

    pts_ref, _ = get_triangular_lattice(nx, ny)
    pts_ref = (pts_ref.T - pts_ref.mean(axis=1)).T
    x_max, y_max = pts_ref.max(axis=1)
    x_min, y_min = pts_ref.min(axis=1)
    x_max += .1; y_max += .1; x_min -= .1; y_min -= .1;

    pts_masked = pts[:, (x_min<=pts[0])&(pts[0]<=x_max)&(y_min<=pts[1])&(pts[1]<=y_max)]
    
    tri = spatial.Delaunay(pts_masked.T)
    # remove simplices with very long edges, which can occur at boundary
    max_lengths = [np.linalg.norm(pts_masked[:,x]-pts_masked[:,np.roll(x, 1)], axis=0).max()
                   for x in tri.simplices]
    simplices = [s for s, l in zip(tri.simplices, max_lengths) if l < max_l]
    pre_mesh = ListOfVerticesAndFaces(tri.points, simplices)
    mesh = HalfEdgeMesh(pre_mesh)

    mesh.transform_vertices(lambda v: v+np.random.normal(scale=noise))
    
    return mesh

# %% ../04b_boundary_conditions_jax.ipynb 23
@jit
def polygon_area(pts):
    """area of polygon assuming no self-intersection. pts.shape (n_vertices, 2)"""
    return jnp.sum(pts[:,0]*jnp.roll(pts[:,1], 1, axis=0) - jnp.roll(pts[:,0], 1, axis=0)*pts[:,1], axis=0)/2

@jit
def polygon_perimeter(pts):
    """perimeter of polygon assuming no self-intersection. pts.shape (n_vertices, 2)"""
    return jnp.sum(jnp.linalg.norm(pts-jnp.roll(pts, 1, axis=0), axis=1), axis=0)

# %% ../04b_boundary_conditions_jax.ipynb 24
@patch
def get_areas(self: HalfEdgeMesh):
    area_dict = {}
    for v in self.vertices.values():
        nghbs = v.get_face_neighbors()
        if (None in nghbs):
            area_dict[v._vid] = np.nan
        else: 
            area_dict[v._vid] = polygon_area(np.stack([fc.dual_coords for fc in nghbs]))
    return area_dict

# %% ../04b_boundary_conditions_jax.ipynb 25
@patch
def get_primal_energy_fct_jax(self: HalfEdgeMesh, bdry_list=None):
    """Get arrays necessary to compute primal energy from primal vertices. Cell based shape tensor.
    bdry_list: [(penalty function, vertex ids),]
    """

    # book-keeping
    face_keys = sorted(self.faces.keys())
    face_key_dict = {key: ix for ix, key in enumerate(sorted(self.faces.keys()))}
    n_faces = len(self.faces)
    
    # stuff for boundary energy
    bdry_list = [] if bdry_list is None else bdry_list
    bdry_list = [bdry + [[]] for bdry in bdry_list] #  3rd entry is for the cell ids 
    
    # stuff for the shape tensor energy
    cell_list = []
    rest_shapes = []
    # for future "convenience" also return a vector of _vids corresponding to the cell list
    cell_list_vids = []
    for v in self.vertices.values():    # iterate around vertex.
        neighbors = v.get_face_neighbors()
        if not (None in neighbors):
            cell = jnp.array([face_key_dict[fc._fid] for fc in neighbors])
            cell_list.append(cell)
            cell_list_vids.append(v._vid)
            # check if the cell is in any bdry:
            for bdry in bdry_list:
                if v._vid in bdry[1]:
                    bdry[2].append(len(cell_list)-1)
            
            rest_shapes.append(v.rest_shape)
    valences = [len(cell) for cell in cell_list]
    max_valence = max(valences)
    valence_mask = jnp.array([x*[1,]+(max_valence-x)*[0,] for x in valences])
    # valence mask = (n_cells, max_valence). entry for each cell indicates whether a vertex is a duplicate
    cell_list = jnp.array([jnp.pad(cell, (0, max_valence-len(cell)), mode="edge") for cell in cell_list])
    rest_shapes = jnp.stack(rest_shapes)
    bdry_list = [[bdry[0], jnp.array(bdry[2])] for bdry in bdry_list]

    # stuff for the angle penalty
    e_dual = [] # dual vertices do not move during optimization, so collect the actual edges
    e_lst_primal = [] # for primal, collect the indices

    for he in self.hes.values():
        if (he.face is not None) and (he.twin.face is not None) and he.duplicate:
            dual_edge = he.vertices[1].coords-he.vertices[0].coords
            # rotate by 90 degrees
            dual_edge = jnp.array([dual_edge[1], -dual_edge[0]])
            dual_edge = dual_edge / np.linalg.norm(dual_edge)
            primal_edge = [face_key_dict[fc._fid] for fc in [he.face, he.twin.face]] # 0= he, 1= twin
            e_dual.append(dual_edge)
            e_lst_primal.append(primal_edge)
    e_dual = jnp.array(e_dual)
    e_lst_primal = jnp.array(e_lst_primal)        
    
    return (e_lst_primal, e_dual, cell_list, rest_shapes, bdry_list, valence_mask), np.array(cell_list_vids)

# %% ../04b_boundary_conditions_jax.ipynb 28
@jit
def get_E(x0, e_lst_primal, e_dual, cell_list, rest_shapes, bdry_list, valence_mask,
          mod_bulk=1, mod_shear=.1, angle_penalty=100, bdry_penalty=10, epsilon_l=1e-3,
          A0=jnp.sqrt(3)/2, mod_area=0):
    n_faces = int(x0.shape[0]/2)
    x, y = (x0[:n_faces], x0[n_faces:])
    pts = jnp.stack([x, y], axis=-1)
    
    # face-based shape energy
    cells = jnp.stack([pts[i] for i in cell_list.T], axis=0)
    edges = cells - jnp.roll(cells, 1, axis=0)
    lengths = jnp.sqrt(jnp.sum(edges**2, axis=-1)+epsilon_l**2)
    # + epsilon**2 to avoid non-differentiable sqrt at 0-length edges (occur due to padding)
    units = (edges.T/lengths.T).T
    tensors = jnp.einsum('efi,efj->fij', edges, units)
    delta = tensors - rest_shapes
    E_shape = jnp.mean(mod_shear*jnp.sum(delta**2, axis=(1,2)) + 
                       mod_bulk*(delta[:,0,0]+delta[:,1,1])**2)

    # angle penalty
    e_primal = pts[e_lst_primal[:,1],:] - pts[e_lst_primal[:,0],:] # he.twin.face-he.face
    lengths = jnp.sqrt(jnp.sum(e_primal**2, axis=-1)+epsilon_l**2)
    # + epsilon to avoid 0-division error and make penalty smooth as length passes through 0
    E_angle = angle_penalty*jnp.mean(1-jnp.einsum('ei,ei->e', e_primal, e_dual)/lengths)
    
    # boundary conditions
    E_bdry = 0
    for bdry in bdry_list:
        centroids = (jnp.sum(cells[:,bdry[1]].T*valence_mask[bdry[1]], axis=-1)
                     /jnp.sum(valence_mask[bdry[1]],axis=-1))
        E_bdry = E_bdry + bdry_penalty*jnp.sum(bdry[0](centroids)) # shape (2, n_cells_in_bdry)
        
    # add area penalty - actually makes things worse!
    areas = polygon_area(cells.transpose((0,2,1)))
    E_area = jnp.mean(mod_area*(areas-A0)**2)
    
    return E_angle + E_bdry + E_area + E_shape

get_E_jac = jit(jgrad(get_E))
#get_E_jac = jgrad(get_E)

# %% ../04b_boundary_conditions_jax.ipynb 42
@patch
def get_tri_areas(self: HalfEdgeMesh):
    area_dict = {}
    for key, fc in self.faces.items():
        area_dict[key] = float(polygon_area(np.stack([he.vertices[0].coords for he in fc.hes][::-1])))
    return area_dict

# %% ../04b_boundary_conditions_jax.ipynb 43
@patch
def get_dual_energy_fct_jax(self: HalfEdgeMesh):
    """Get arrays for triangulation flattening. JAX style."""
    e_lst = []
    tri_lst = []
    rest_lengths = []

    # we will need to look up which vertex key corresponds to list position
    vertex_key_dict = {key: ix for ix, key in enumerate(sorted(self.vertices.keys()))}
    
    for e in self.hes.values():
        if e.duplicate: # avoid duplicates
            e_lst.append([vertex_key_dict[v._vid] for v in e.vertices])
            rest_lengths.append((e.rest+e.twin.rest)/2)
    e_lst = jnp.array(e_lst).T
    rest_lengths = jnp.array(rest_lengths)
    
    for fc in self.faces.values():
        tri_lst.append([vertex_key_dict[he.vertices[0]._vid] for he in fc.hes][::-1])
    tri_lst = jnp.array(tri_lst).T
    n_vertices = len(self.vertices)
    
    return e_lst, rest_lengths, tri_lst, n_vertices

# %% ../04b_boundary_conditions_jax.ipynb 44
@Partial(jit, static_argnums=(4,))
def get_E_dual(x0, e_lst, rest_lengths, tri_lst, n_vertices, mod_area=0.01, A0=jnp.sqrt(3)/4):
    """Dual energy function for triangulation flattening"""
    pts = jnp.stack([x0[:n_vertices], x0[n_vertices:]], axis=1)
    lengths = jnp.linalg.norm(pts[e_lst[0]]-pts[e_lst[1]], axis=1)
    
    E_length = 1/2 * jnp.sum((lengths-rest_lengths)**2)
    # triangle area penalty
    A = polygon_area(pts[tri_lst].transpose((0,2,1)))
    # orientation penalty:
    E_area = mod_area/2 *(100*jnp.sum(jrelu(-A+A0/4)**2) + jnp.sum((A-A0)**2))
    # relu term penalizes 'flipped' triangles with incorrect orientation.
    
    return E_length + E_area
  
get_E_dual_jac = jit(jgrad(get_E_dual), static_argnums=(4,))

# %% ../04b_boundary_conditions_jax.ipynb 45
@patch
def flatten_triangulation_jax(self: HalfEdgeMesh, tol=1e-4, verbose=True, mod_area=0, A0=jnp.sqrt(3)/4):
    """Flatten triangulation"""
    energy_arrays = self.get_dual_energy_fct_jax()
    x0 = self.vertices_to_initial_cond()
    sol = optimize.minimize(get_E_dual, x0, method="CG", jac=get_E_dual_jac, tol=tol, # CG, BFGS
                            args=energy_arrays+(mod_area, A0))
    if sol["status"] !=0 and verbose:
        print("Triangulation optimization failed")
        print(sol["message"])
    new_coord_dict = self.initial_cond_to_vertices(sol["x"])
    for key, val in self.vertices.items():
        val.coords = new_coord_dict[key]
    self.set_rest_lengths()

# %% ../04b_boundary_conditions_jax.ipynb 49
@patch
def get_bdry(self: HalfEdgeMesh):
    """get boundary hes. Works if boundary is simply connected. Could be modified for general case"""
    bdry = []
    start = next(he for he in self.hes.values() if he.face is None)
    he = start
    returned = False
    while not returned:
        bdry.append(he)
        he = he.nxt
        returned = (he==start)
    return bdry

def get_conformal_transform(mesh1, mesh2):
    """Get rotation+scaling+translation to match mesh2's triangulation to mesh1. Preserves overall area"""
    bdry1 = np.stack([he.vertices[0].coords for he in mesh1.get_bdry()])
    bdry2 = np.stack([he.vertices[0].coords for he in mesh2.get_bdry()])
    rescale = np.sqrt(polygon_area(bdry1)/polygon_area(bdry2))

    pts1 = np.stack([v.coords for v in mesh1.vertices.values()])
    pts2 = np.stack([v.coords for v in mesh2.vertices.values()])
    mean1, mean2 = (pts1.mean(axis=0), pts2.mean(axis=0))

    rotation = linalg.orthogonal_procrustes(rescale*(pts2-mean2), pts1-mean1)[0]

    return lambda x: rotation.T@(rescale*(x-mean2))+mean1


# %% ../04b_boundary_conditions_jax.ipynb 53
def excitable_dt_act_pass(Ts, Tps, k=1, m=2, k3=.2):
    """Time derivative of tensions under excitable tension model with constrained area,
    with passive tension for post intercalation. Variant: completely deactivate feedback for m=1.
    k3 is a cutoff in the excitable tension dynamics, for numerical stability at the mesh edges.
    """
    dT_dt = (m-1)*((Ts-Tps)**m - k3*(Ts-Tps)**(m+1) - k*Tps) - k*(m==1) * (Ts-1)
    # use relative tension
    #T_mean = Ts.mean()
    #dT_dt = T_mean *((m-1)*(((Ts-Tps)/T_mean)**m - k3*((Ts-Tps)/T_mean)**3 - k*Tps/T_mean ) )
    
    dTp_dt = -k*Tps
    area_jac = sides_area_jac(Ts-Tps)
    area_jac /= np.linalg.norm(area_jac)
    dT_dt -= area_jac * (area_jac@dT_dt)    
    return dT_dt, dTp_dt

# %% ../04b_boundary_conditions_jax.ipynb 54
@patch
def euler_step(self: HalfEdgeMesh, dt=.005, rhs=excitable_dt_post, params=None,
               rhs_rest_shape=None):
    """RHS: callable Ts, Tps -> dTs_dt, dTps_dt. Params can either be a dict of keyword args
    to the RHS function, or a callable faceid -> keyword dict.
    rhs_rest_shape: v -> d_rest_shape_dt, for rest shape dynamics (e.g. viscous relaxation)
    """
    rhs_rest_shape = (lambda v: 0) if rhs_rest_shape is None else rhs_rest_shape
    for fc in self.faces.values():
        # collect edges
        Ts, Tps = (np.array([he.rest for he in fc.hes]), np.array([he.passive for he in fc.hes]))
        if isinstance(params, dict):
            dT_dt, dTp_dt = rhs(Ts, Tps, **params)
        elif callable(params):
            dT_dt, dTp_dt = rhs(Ts, Tps, **params(fc._fid))
        Ts += dt*dT_dt
        Tps += dt*dTp_dt
        for T, Tp, he in zip(Ts, Tps, fc.hes):
            he.rest = T
            he.passive = Tp
    for v in self.vertices.values():
        v.rest_shape += dt*rhs_rest_shape(v)


# %% ../04b_boundary_conditions_jax.ipynb 55
@patch
def flatten_triangulation(self: HalfEdgeMesh, tol=1e-3, verbose=True, reg_A=0, A0=sqrt(3)/4):
    """Flatten triangulation"""
    get_E, grd = self.get_energy_fct(reg_A=0, A0=A0)
    x0 = self.vertices_to_initial_cond()
    sol = optimize.minimize(get_E, x0, method="CG", jac=grd, tol=tol)
    if sol["status"] !=0 and verbose:
        print("Triangulation optimization failed")
        print(sol["message"])
    new_coord_dict = self.initial_cond_to_vertices(sol["x"])
    for key, val in self.vertices.items():
        val.coords = new_coord_dict[key]
    self.set_rest_lengths()


# %% ../04b_boundary_conditions_jax.ipynb 56
@patch
def optimize_cell_shape(self: HalfEdgeMesh, bdry_list=None,
                        energy_args=None, cell_id_to_modulus=None,
                        tol=1e-3, maxiter=500, verbose=True, bdr_weight=2):
    """Primal optimization. cell_id_to_modulus: function from _vid to relative elastic modulus"""
    x0 = self.dual_vertices_to_initial_cond()
    get_E_arrays, cell_list_vids = self.get_primal_energy_fct_jax(bdry_list)

    if energy_args is None:
        energy_args = {"mod_bulk": 1, "mod_shear": .2,"angle_penalty": 1000, "bdry_penalty": 100,
                       "epsilon_l": 1e-4, "A0": jnp.sqrt(3)/2, "mod_area": 0}
    if cell_id_to_modulus is not None:
        mod_bulk = energy_args["mod_bulk"]*np.vectorize(cell_id_to_modulus)(cell_list_vids)
        mod_shear = energy_args["mod_shear"]*np.vectorize(cell_id_to_modulus)(cell_list_vids)
        mod_area = energy_args["mod_area"]*np.vectorize(cell_id_to_modulus)(cell_list_vids)
    else:
        mod_bulk, mod_shear, mod_area = (energy_args["mod_bulk"], energy_args["mod_shear"],
                                         energy_args["mod_area"])
    
    if bdr_weight != 1:
        is_bdr = np.array([any([fc.is_bdr() for fc in self.vertices[v].get_face_neighbors()])
                          for v in cell_list_vids])
        mod_bulk *= (bdr_weight*is_bdr+(1-is_bdr))
        mod_shear *= (bdr_weight*is_bdr+(1-is_bdr))
        mod_area *= (bdr_weight*is_bdr+(1-is_bdr))

    
    cell_shape_args = (mod_bulk, mod_shear, energy_args["angle_penalty"], energy_args["bdry_penalty"],
                       energy_args["epsilon_l"], energy_args["A0"], energy_args["mod_area"])
    
    sol = optimize.minimize(get_E, x0, jac=get_E_jac, args=get_E_arrays+cell_shape_args,
                             method="CG", tol=tol, options={"maxiter": maxiter})
    if sol["status"] !=0 and verbose:
        print("Cell shape optimization failed", sol["message"])
    new_coord_dict = self.initial_cond_to_dual_vertices(sol["x"])
    for key, val in self.faces.items():
        val.dual_coords = new_coord_dict[key]

# %% ../04b_boundary_conditions_jax.ipynb 57
def get_flip_edge(msh, minimal_l, exclude):
    primal_lengths = msh.get_primal_edge_lens(oriented=True)
    primal_lengths = sorted(primal_lengths.items(), key=lambda x: x[1])
    primal_lengths = [x for x in primal_lengths if (x[1] <= minimal_l)]
    primal_lengths = [x for x in primal_lengths if not x[0] in exclude]
    if primal_lengths:
        return primal_lengths[0][0]
    return None
