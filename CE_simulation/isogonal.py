# AUTOGENERATED! DO NOT EDIT! File to edit: ../03_real_shape_optimization.ipynb.

# %% auto 0
__all__ = ['polygon_area', 'polygon_perimeter', 'get_vertex_energy', 'is_convex_polygon', 'rotate_about_center']

# %% ../03_real_shape_optimization.ipynb 4
from .triangle import *
from .tension import *
from .delaunay import *

# %% ../03_real_shape_optimization.ipynb 5
import os
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
import ipywidgets as widgets
from matplotlib import animation, rc
rc('animation', html='html5')

# %% ../03_real_shape_optimization.ipynb 8
import autograd.numpy as anp  # Thinly-wrapped numpy
from autograd import grad as agrad

# %% ../03_real_shape_optimization.ipynb 14
@patch
def get_shape_tensor(self: Face):
    neighbors = [he.twin.face for he in self.hes]
    if (None in neighbors) or self is None:
        return np.eye(2)
    edges = np.stack([x.dual_coords-self.dual_coords for x in neighbors])
    lengths = np.linalg.norm(edges, axis=-1)
    return 2*np.einsum('ei,ej->ij', edges, (edges.T/(lengths.T+1e-3)).T)

@patch
def set_rest_shapes(self: HalfEdgeMesh):
    """ill defined for boundary faces."""
    for fc in self.faces.values():
        if fc is not None:
            fc.rest_shape = fc.get_shape_tensor()

# %% ../03_real_shape_optimization.ipynb 16
def polygon_area(pts):
    """area of polygon assuming no self-intersection. pts.shape (n_vertices, 2)"""
    return anp.sum(pts[:,0]*anp.roll(pts[:,1], 1, axis=0) - anp.roll(pts[:,0], 1, axis=0)*pts[:,1], axis=0)/2

def polygon_perimeter(pts):
    """perimeter of polygon assuming no self-intersection. pts.shape (n_vertices, 2)"""
    return anp.sum(anp.linalg.norm(pts-anp.roll(pts, 1, axis=0), axis=1))

def get_vertex_energy(pts, A0=1, P0=1, mod_shear=0, mod_bulk=1):
    """Get vertex style energy"""
    return mod_bulk*(polygon_area(pts)-A0)**2 + mod_shear*(polygon_perimeter(pts)-P0)**2

@patch
def get_area(self: Vertex):
    neighbors = [fc for fc in self.get_face_neighbors()]
    if None in neighbors:
        return None
    return polygon_area(np.stack([fc.dual_coords for fc in neighbors]))

# %% ../03_real_shape_optimization.ipynb 23
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


# %% ../03_real_shape_optimization.ipynb 25
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

# %% ../03_real_shape_optimization.ipynb 29
@patch
def get_primal_energy_fct_vertices(self: HalfEdgeMesh, mod_bulk=1, mod_shear=.01, angle_penalty=100,
                                   reg_bulk=0, A0=sqrt(3)/2,  max_valence=10, epsilon_l=1e-5):
    """Get function to compute primal energy from primal vertices."""

    # stuff for the shape tensor energy
    face_list = []    
    face_key_dict = {key: ix for ix, key in enumerate(sorted(self.faces.keys()))}
    face_key_dict[None] = None
    rest_shapes = []
    for fc in self.faces.values():
        neighbors = [he.twin.face for he in fc.hes]
        if not (None in neighbors):
            face_list.append(anp.array([face_key_dict[x._fid] for x in neighbors]
                                      +[face_key_dict[fc._fid]]))
            rest_shapes.append(fc.rest_shape)
    face_list = anp.array(face_list).T
    rest_shapes = anp.stack(rest_shapes)
    n_faces = len(self.faces)

    # stuff for the vertex-energy-based regularization
    if reg_bulk > 0:
        cell_list = []
        for v in self.vertices.values():    # iterate around vertex.
            neighbors = v.get_face_neighbors()
            if not (None in neighbors):
                cell = [face_key_dict[fc._fid] for fc in neighbors]
                try:
                    cell = anp.pad(cell, (0, max_valence-len(cell)), mode="edge")
                except ValueError:
                    print(f"cell with more than {max_valence} nghbs, increase max_valence")
                cell_list.append(cell)
        cell_list = anp.array(cell_list)
    
    # stuff for the angle penalty
    e_dual = [] # dual vertices do not move during optimization, so collect the actual edges
    e_lst_primal = [] # for primal, collect the indices

    for he in self.hes.values():
        if (he.face is not None) and (he.twin.face is not None) and he.duplicate:
            dual_edge = he.vertices[1].coords-he.vertices[0].coords
            # rotate by 90 degrees
            dual_edge = anp.array([dual_edge[1], -dual_edge[0]])
            dual_edge = dual_edge / np.linalg.norm(dual_edge)
            primal_edge = [face_key_dict[fc._fid] for fc in [he.face, he.twin.face]] # 0= he, 1= twin
            e_dual.append(dual_edge)
            e_lst_primal.append(primal_edge)
    e_dual = anp.array(e_dual)
    e_lst_primal = anp.array(e_lst_primal)
    
    # breaking translational invariance.
    center = anp.mean([v.coords for v in self.vertices.values()], axis=0)
    
    def get_E(x0):
        x, y = (x0[:n_faces], x0[n_faces:])
        pts = anp.stack([x, y], axis=-1)
        # shape energy
        edges = anp.stack([pts[a]-pts[face_list[3]] for a in face_list[:3]])
        lengths = anp.linalg.norm(edges, axis=-1) + 10*epsilon_l # + epsilon to avoid 0-division error
        units = (edges.T/lengths.T).T        
        tensors = 2*anp.einsum('efi,efj->fij', edges, units) - rest_shapes
        E_shape = (mod_shear*4*anp.mean(tensors**2)
                   + mod_bulk*anp.mean((tensors[:,0,0]+tensors[:,1,1])**2))
        # regularize with the vertex model energy
        if reg_bulk > 0:
            poly = anp.stack([[x[i], y[i]] for i in cell_list.T]) # shape (max_valence, 2, n_cells)
            E_vertex = reg_bulk*anp.mean((polygon_area(poly)-A0)**2)
        else:
            E_vertex = 0
        # angle penalty
        e_primal = pts[e_lst_primal[:,1]] - pts[e_lst_primal[:,0]] # he.twin.face-he.face
        lengths = anp.linalg.norm(e_primal, axis=-1) 
        # + epsilon to avoid 0-division error and make penalty smooth as length passes through 0
        penalty = (1-anp.einsum('ei,ei->e', e_primal, e_dual)/(lengths+epsilon_l))
        # makes energy barrier so need epsilon small!
        #penalty = (lengths/(lengths+epsilon_l)-anp.einsum('ei,ei->e', e_primal, e_dual)/(lengths+epsilon_l))
        #penalty = (lengths-anp.einsum('ei,ei->e', e_primal, e_dual))
        #penalty = (lengths-anp.einsum('ei,ei->e', e_primal, e_dual)) / (lengths+1e-2)
        # all of the above lead to precision loss error

        # this ensures that there is no energy barrier towards decreasing edge length at fixed angle
        # alternative: e_primal = (e_primal.T/lengths.T).T; and *anp.tanh(lengths/1e-2) 
        E_angle = angle_penalty * anp.mean(penalty) #**2
        # break translation symmetry
        E_trans = 1/2*((anp.mean(x)-center[0])**2+(anp.mean(y)-center[0]))**2
        
        return  E_angle + E_shape + E_vertex + E_trans
    
    return get_E, agrad(get_E)

# %% ../03_real_shape_optimization.ipynb 44
@patch
def get_primal_edge_lens(self: HalfEdgeMesh, oriented=True):
    len_dict = {}
    for he in self.hes.values():
        if (he.face is not None) and (he.twin.face is not None) and he.duplicate:
            primal_vec = he.face.dual_coords-he.twin.face.dual_coords
            length = np.linalg.norm(primal_vec)
            if oriented:
                centroid_vec = (np.mean([x.vertices[0].coords for x in he.face.hes], axis=0)
                                -np.mean([x.vertices[0].coords for x in he.twin.face.hes], axis=0))
                length *= np.sign(np.dot(primal_vec, centroid_vec))
            len_dict[he._heid] = length
    return len_dict

# %% ../03_real_shape_optimization.ipynb 48
def is_convex_polygon(polygon):
    """Return True if the polynomial defined by the sequence of 2D
    points is 'strictly convex': points are valid, side lengths non-
    zero, interior angles are strictly between zero and a straight
    angle, and the polygon does not intersect itself.

    NOTES:  1.  Algorithm: the signed changes of the direction angles
                from one side to the next side must be all positive or
                all negative, and their sum must equal plus-or-minus
                one full turn (2 pi radians). Also check for too few,
                invalid, or repeated points.
            2.  No check is explicitly done for zero internal angles
                (180 degree direction-change angle) as this is covered
                in other ways, including the `n < 3` check.
    Source: stackoverflow.com/questions/471962/how-do-i-efficiently-determine-if-a-polygon-is\
    -convex-non-convex-or-complex
    """
    try:  # needed for any bad points or direction changes
        # Check for too few points
        if polygon.shape[0] < 3:
            return False
        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = np.arctan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        # Check each point (the side ending there, its angle) and accum. angles
        for ndx, newpoint in enumerate(polygon):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = np.arctan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  # repeated consecutive points
            # Calculate & check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -pi:
                angle += 2*pi  # make it in half-open interval (-Pi, Pi]
            elif angle > pi:
                angle -= 2*pi
            if ndx == 0:  # if first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # not both pos. or both neg.
                    return False
            # Accumulate the direction-change angle
            angle_sum += angle
        # Check that the total number of full turns is plus-or-minus 1
        return abs(round(angle_sum / (2*pi))) == 1
    except (ArithmeticError, TypeError, ValueError):
        return False  # any exception means not a proper convex polygon

# %% ../03_real_shape_optimization.ipynb 49
@patch
def find_primal_problematic(self: HalfEdgeMesh):
    """Identify problematic cells and edges"""
    bad_edges = [key for key, val in self.get_primal_edge_lens(oriented=True).items() if val <0]
    bad_cells = []
    for v in self.vertices.values():
        neighbors = v.get_face_neighbors()
        if not (None in neighbors):
            if not is_convex_polygon(np.stack([fc.dual_coords for fc in neighbors])):
                bad_cells.append(v._vid)
    return bad_cells, bad_edges

# %% ../03_real_shape_optimization.ipynb 53
# for re-setting primal vertex positions after intercalation
def rotate_about_center(x, angle=pi/2):
    """Rotate pts about center. x.shape = (n_pts, 2)"""
    center = np.mean(x, axis=0)
    return (x-center)@rot_mat(angle)+np.mean(x, axis=0)

# %% ../03_real_shape_optimization.ipynb 54
@patch
def optimize_cell_shape(self: HalfEdgeMesh, energy_fct_kwargs=None, tol=1e-3, maxiter=250, verbose=True):
    """primal optimization"""
    energy_fct_kwargs = dict() if energy_fct_kwargs is None else energy_fct_kwargs
    get_E, jac = self.get_primal_energy_fct_vertices(**energy_fct_kwargs)
    x0 = self.dual_vertices_to_initial_cond()
    sol = optimize.minimize(get_E, x0, jac=jac, method="BFGS", tol=tol, options={"maxiter": maxiter})
    if sol["status"] !=0 and verbose:
        print("Cell shape optimization failed", sol["message"])
    new_coord_dict = self.initial_cond_to_dual_vertices(sol["x"])
    for key, val in self.faces.items():
        val.dual_coords = new_coord_dict[key]
