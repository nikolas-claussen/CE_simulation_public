# AUTOGENERATED! DO NOT EDIT! File to edit: ../03_real_shape_optimization.ipynb.

# %% auto 0
__all__ = ['get_shape_tensor', 'get_triangle_shape_tensor', 'get_shape_energy', 'polygon_area', 'polygon_perimeter',
           'get_vertex_energy', 'rotate_about_center']

# %% ../03_real_shape_optimization.ipynb 4
from .triangle import *
from .tension import *
from .delaunay import *

# %% ../03_real_shape_optimization.ipynb 5
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from numpy import sin, cos, tan, pi, sqrt, arccos, arctan, arctan2
from numpy.linalg import norm

from scipy.integrate import solve_ivp
from scipy import ndimage
from scipy import spatial
from scipy import optimize

from tqdm.notebook import tqdm

from math import floor, ceil

import sys

from copy import deepcopy

from collections import Counter

# %% ../03_real_shape_optimization.ipynb 6
from dataclasses import dataclass
from typing import Union, Dict, List, Tuple, Iterable, Callable
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../03_real_shape_optimization.ipynb 7
import autograd.numpy as anp  # Thinly-wrapped numpy
from autograd import grad as agrad

from scipy.sparse import csc_matrix

# %% ../03_real_shape_optimization.ipynb 10
def get_shape_tensor(poly: NDArray[Shape["N, 2"], Float], metric=False):
    """Shape tensor. normalized so that isotropic shape with unit edge length is identity"""
    edges = poly - anp.roll(poly, 1, axis=0)
    if metric: # don't normalize
        return 2 * anp.mean(anp.einsum("ei,ej->ije", edges, edges), axis=-1)
    lengths = anp.linalg.norm(edges, axis=-1)
    tensor = 2*anp.mean(anp.einsum("ei,ej->ije", edges, edges)/lengths, axis=-1)
    return tensor

def get_triangle_shape_tensor(poly: NDArray[Shape["N, 2"], Float],):
    
    center = anp.mean(poly, axis=0)
    triangles = anp.stack([np.stack(poly.shape[0]*[center]), poly, anp.roll(poly, 1, axis=0)], axis=1)
    signed_areas = anp.sum(triangles[...,0]*anp.roll(triangles[...,1], 1, axis=1)
                          - anp.roll(triangles[...,0], 1, axis=1)*triangles[...,1], axis=1)/2
    orientations = signed_areas / anp.abs(signed_areas)
    triangle_edges = triangles - anp.roll(triangles, 1, axis=1)
    
    tensor = anp.mean(anp.einsum("vei, vej->ijv", triangle_edges, triangle_edges)*orientations, axis=-1)
    return 2/3*tensor

def get_shape_energy(tensor, rest_shape=np.eye(2), mod_shear=0, mod_bulk=1):
    delta = tensor-rest_shape
    return mod_shear*anp.sum(delta**2)+mod_bulk*anp.trace(delta)**2

# %% ../03_real_shape_optimization.ipynb 14
def polygon_area(pts):
    """area of polygon assuming no self-intersection. pts.shape (n_vertices, 2)"""
    return anp.sum(pts[:,0]*anp.roll(pts[:,1], 1, axis=0) - anp.roll(pts[:,0], 1, axis=0)*pts[:,1])/2

def polygon_perimeter(pts):
    """perimeter of polygon assuming no self-intersection. pts.shape (n_vertices, 2)"""
    return anp.sum(anp.linalg.norm(pts-anp.roll(pts, 1, axis=0), axis=1))

def get_vertex_energy(pts, A0=1, P0=1, mod_shear=0, mod_bulk=1):
    """Get vertex style energy"""
    return mod_bulk*(polygon_area(pts)-A0)**2 + mod_shear*(polygon_perimeter(pts)-P0)**2

# %% ../03_real_shape_optimization.ipynb 25
# new plotting functions
@patch
def cellplot(self: HalfEdgeMesh, alpha=1):
    """Plot based on primal positions. Might be slow because loops over faces"""
    for fc in self.faces.values():
        for he in fc.hes:
            nghb = he.twin.face
            if nghb is not None:
                line = np.stack([fc.dual_coords, nghb.dual_coords])
                plt.plot(*line.T, c="k", alpha=alpha)

@patch
def labelplot(self: HalfEdgeMesh, vertex_labels=True, face_labels=True,
                     halfedge_labels=False, cell_labels=False):
    """for debugging purposes, a fct to plot a trimesh with labels attached"""
    if face_labels:
        for fc in self.faces.values():
            centroid = np.mean([he.vertices[0].coords for he in fc.hes], axis=0)
            plt.text(*centroid, str(fc._fid), color="k")
    if vertex_labels:
        for v in self.vertices.values():
            plt.text(*(v.coords+np.array([0,.05])), str(v._vid),
                     color="tab:blue", ha="center")
    if cell_labels:
        for v in self.vertices.values():
            nghbs = v.get_face_neighbors()
            if not (None in nghbs):
                center = np.mean([fc.dual_coords for fc in nghbs], axis=0)
                plt.text(*(center), str(v._vid),
                         color="tab:blue", ha="center")
    if halfedge_labels:
        for he in self.hes.values():
            if he.duplicate:
                centroid = np.mean([v.coords for v in he.vertices], axis=0)
                plt.text(*centroid, str(he._heid), color="tab:orange")

# %% ../03_real_shape_optimization.ipynb 26
@patch
def get_face_neighbors(self: Vertex):
    """Get face neighbors of vertex"""
    neighbors = []
    start_he = self.incident[0]
    he = start_he
    returned = False
    while not returned:
        neighbors.append(he.face)
        he = he.nxt.twin
        returned = (he == start_he)
    return neighbors

@patch
def set_centroid(self: HalfEdgeMesh):
    """Set dual positions to triangle centroid"""
    for fc in self.faces.values():
        vecs = []
        returned = False
        start_he = fc.hes[0]
        he = start_he
        while not returned:
            vecs.append(he.vertices[0].coords)
            he = he.nxt
            returned = (he == start_he)
        fc.dual_coords = np.mean(vecs, axis=0)

@patch
def transform_dual_vertices(self: HalfEdgeMesh, trafo: Union[Callable, NDArray[Shape["2, 2"], Float]]):
    for fc in self.faces.values():
        if isinstance(trafo, Callable):
            fc.dual_coords = trafo(fc.dual_coords)
        else:
            fc.dual_coords = trafo.dot(fc.dual_coords)

# %% ../03_real_shape_optimization.ipynb 29
@patch
def get_shape_tensors(self: HalfEdgeMesh):
    """Get current shape tensors as dict"""
    # iterate around vertex.
    result_dict = {}
    for v in self.vertices.values():
        neighbors = v.get_face_neighbors()
        if not (None in neighbors):
            polygon = np.stack([fc.dual_coords for fc in neighbors])
            result_dict[v._vid] = get_triangle_shape_tensor(polygon)
        else: # leave at default value
            result_dict[v._vid] = np.eye(2)
    return result_dict

@patch
def set_rest_shapes(self: HalfEdgeMesh):
    """Set rest shapes to current shapes"""
    shape_dict = self.get_shape_tensors()
    for v in self.vertices.values():
        v.rest_shape = shape_dict[v._vid]

# %% ../03_real_shape_optimization.ipynb 30
@patch
def get_energies(self: HalfEdgeMesh, mod_shear=0, mod_bulk=1, A0=1, P0=1, energy="shape"):
    """Does not set energy of boundary vertices. energy can be shape or vertex"""
    energy_dict = {}
    tensors = mesh.get_shape_tensors()
    for key, val in tensors.items():
        if not (None in self.vertices[key].get_face_neighbors()):
            if energy =="shape":
                energy_dict[key] = get_shape_energy(val, rest_shape=mesh.vertices[key].rest_shape,
                                                    mod_shear=mod_shear, mod_bulk=mod_bulk)
            elif energy == "vertex":
                pts = np.stack([fc.dual_coords for fc in self.vertices[key].get_face_neighbors()])
                energy_dict[key] = get_vertex_energy(pts, A0=A0, P0=P0, mod_shear=mod_shear, mod_bulk=mod_bulk)
    return energy_dict

# %% ../03_real_shape_optimization.ipynb 32
@patch
def dual_vertices_to_initial_cond(self: HalfEdgeMesh):
    """Format dual vertices for use in energy minimization."""
    face_keys = sorted(self.faces.keys())
    dual_vertex_vector = np.stack([self.faces[key].dual_coords for key in face_keys]).T
    return np.hstack([dual_vertex_vector[0], dual_vertex_vector[1]])
       
@patch
def initial_cond_to_dual_vertices(self: HalfEdgeMesh, x0):
    """Reverse of format dual vertices for use in energy minimization."""
    face_keys = sorted(self.faces.keys())
    x, y = (x0[:int(len(x0)/2)], x0[int(len(x0)/2):])
    dual_vertex_vector = np.stack([x, y], axis=1)
    return {key: val for key, val in zip(face_keys, dual_vertex_vector)}


# %% ../03_real_shape_optimization.ipynb 37
@patch
def get_angle_deviation(self: HalfEdgeMesh):
    """Angle between primal and dual edges. For diagnostics"""
    angle_deviation = {}

    for he in self.hes.values():
        if (he.face is not None) and (he.twin.face is not None):
            dual_edge = he.vertices[1].coords-he.vertices[0].coords
            primal_edge = he.face.dual_coords - he.twin.face.dual_coords
            dual_edge = dual_edge / np.linalg.norm(dual_edge)
            primal_edge = primal_edge / np.linalg.norm(primal_edge)        
            angle_deviation[he._heid] = np.dot(dual_edge, primal_edge)**2
    return angle_deviation

# %% ../03_real_shape_optimization.ipynb 40
@patch
def get_primal_energy_fct(self: HalfEdgeMesh, mod_bulk=1, mod_shear=1e-3, angle_penalty=1e2,
                          A0=1, P0=6, energy="shape"):
    """Get function to compute primal energy from primal vertices."""
    
    # stuff for the shape tensor energy
    primal_face_list = []
    rest_shapes = []
    
    face_key_dict = {key: ix for ix, key in enumerate(sorted(self.faces.keys()))}
    face_key_dict[None] = None
    
    for v in self.vertices.values():    # iterate around vertex.
        neighbors = v.get_face_neighbors()
        if not (None in neighbors):
            primal_face_list.append(anp.array([fc._fid for fc in neighbors]))
            rest_shapes.append(v.rest_shape)
    # cells might have differing #vertices, so don't make primal face list into array
    rest_shapes = anp.array(rest_shapes)
    n_faces = len(self.faces)

    # stuff for the angle penalty
    e_dual = [] # dual vertices do not move during optiomization, so collect the actual edges
    e_lst_primal = [] # for primal, collect the indices

    for he in self.hes.values():
        if (he.face is not None) and (he.twin.face is not None) and he.duplicate:
            dual_edge = he.vertices[1].coords-he.vertices[0].coords
            dual_edge = dual_edge / np.linalg.norm(dual_edge)
            primal_edge = [face_key_dict[fc._fid] for fc in [he.face, he.twin.face]]
            e_dual.append(dual_edge)
            e_lst_primal.append(primal_edge)
    e_dual = anp.array(e_dual)
    e_lst_primal = anp.array(e_lst_primal)    
    center = anp.mean([fc.dual_coords for fc in self.faces.values()], axis=0)

    def get_E(x0):
        x, y = (x0[:n_faces], x0[n_faces:])
        # shape energy - can be either vertex style or shape tensor based
        #tensors = []
        E_shape = 0
        for fc in primal_face_list:
            pts = anp.stack([x[fc], y[fc]], axis=-1)
            E_shape = E_shape+mod_bulk*(polygon_area(pts)-A0)**2+mod_shear*(polygon_perimeter(pts)-P0)**2
            #tensors.append(get_shape_tensor(pts, metric=True))
        #tensors = anp.array(tensors)
        #delta = tensors - rest_shapes
        #E_shape = anp.mean(mod_bulk*(delta[:,0,0]+delta[:,1,1])**2
        #                 + mod_shear*(delta[:,0,0]**2+2*delta[:,0,1]**2+delta[:,1,1]**2))
        # angle penalty
        pts = anp.stack([x, y], axis=-1)
        e_primal = pts[e_lst_primal[:,1]] - pts[e_lst_primal[:,0]]
        e_primal = (e_primal.T/anp.linalg.norm(e_primal, axis=-1)).T
        E_angle = angle_penalty * anp.mean(anp.einsum('ei,ei->e', e_primal, e_dual)**2)
        # break translation symmetry
        E_trans = 1/2*((anp.mean(x)-center[0])**2+(anp.mean(y)-center[0]))**2
        
        return E_shape + E_angle #+ E_trans
    
    return get_E, agrad(get_E)

# %% ../03_real_shape_optimization.ipynb 55
@patch
def get_primal_edge_lens(self: HalfEdgeMesh):
    return {he._heid: np.linalg.norm(he.face.dual_coords-he.twin.face.dual_coords)
            for key, he in self.hes.items() if (he.face is not None) and (he.twin.face is not None)}
    
    return None

# %% ../03_real_shape_optimization.ipynb 56
def rotate_about_center(x, angle=pi/2):
    """Rotate pts about center. x.shape = (n_pts, 2)"""
    center = np.mean(x, axis=0)
    return (x-center)@rot_mat(angle)+np.mean(x, axis=0)
