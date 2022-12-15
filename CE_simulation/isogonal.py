# AUTOGENERATED! DO NOT EDIT! File to edit: ../03_real_shape_optimization.ipynb.

# %% auto 0
__all__ = ['get_E_jac', 'get_shape_tensor', 'get_shape_energy', 'get_vertex_energy', 'polygon_area', 'get_E',
           'rotate_about_center', 'get_flip_edge']

# %% ../03_real_shape_optimization.ipynb 3
import CE_simulation.mesh as msh
import CE_simulation.tension as tns
import CE_simulation.delaunay as dln

# %% ../03_real_shape_optimization.ipynb 4
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.integrate import solve_ivp
from scipy import ndimage
from scipy import spatial
from scipy import optimize

from tqdm.notebook import tqdm

from copy import deepcopy

from collections import defaultdict

# %% ../03_real_shape_optimization.ipynb 5
from typing import Union, Dict, List, Tuple, Iterable, Callable, Any
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../03_real_shape_optimization.ipynb 6
import jax.numpy as jnp
from jax import grad as jgrad
from jax import jit
from jax.tree_util import Partial
from jax.config import config
from jax.nn import relu as jrelu

config.update("jax_enable_x64", True) # 32 bit leads the optimizer to complain about precision loss
config.update("jax_debug_nans", False)  # useful for debugging, but makes code slower!

# %% ../03_real_shape_optimization.ipynb 7
import ipywidgets as widgets
from matplotlib import animation, rc

# %% ../03_real_shape_optimization.ipynb 10
def get_shape_tensor(poly: NDArray[Shape["*,2,..."],Float], epsilon_l=1e-4) -> NDArray[Shape["2,2,..."],Float]:
    """
    Compute shape tensor from polygon vertex coords.
    
    Assumes that the vertex coords are order clock- or counter-clockwise.
    Computes Sum_e l_e \outer l_e / |l_e| where l_e is the vector along
    polygon e.
    
    Parameters
    ----------
    poly : (n_vertices, 2, n_samples) array
        Input polygon(s)
    epsilon_l : float
        Regularization to keep function diff'ble for 0-edge length polygons
        
    Returns
    -------
    (2, 2, n_samples) array
        Shape tensor
    """

    edges = poly - jnp.roll(poly, 1, axis=0)
    lengths = jnp.sqrt(jnp.sum(edges**2, axis=1)+epsilon_l**2) # to make differentiable
    units = (edges.swapaxes(1,0)/lengths).swapaxes(1,0)
    return jnp.einsum('ei...,ej...->ij...', edges, units)
    
@patch
def get_vrtx_shape_tensor(self: msh.Vertex) -> NDArray[Shape["2,2"],Float]:
    """Get shape tensor for a vertex (=primal cell) using get_shape_tensor. Returns Id for bdry vertices."""
    neighbors = self.get_face_neighbors()
    if (None in neighbors) or self is None:
        return np.eye(2)
    cell = np.stack([fc.dual_coords for fc in neighbors])
    return get_shape_tensor(cell)

@patch
def set_rest_shapes(self: msh.HalfEdgeMesh) -> None:
    """Set rest shape for all mesh vertices to current shape tensor."""
    for v in self.vertices.values():
        v.rest_shape = v.get_shape_tensor()

# %% ../03_real_shape_optimization.ipynb 14
def get_shape_energy(poly: NDArray[Shape["*,2"],Float],
                     rest_shape: NDArray[Shape["2,2"],Float]=jnp.eye(2), A0=jnp.sqrt(3)/2,
                     mod_shear=.5, mod_bulk=1, mod_area=0) -> float:
    """
    Compute shape tensor energy for polygon.
    
    Includes also an optional term for area elasticity.
    
    Parameters
    ----------
    poly: (n_vertices, 2) array
        cell vertex array. assumed to be ordered correctly
    rest_shape: (2, 2) array
        reference shape
    A0: float
        reference area
    mod_shear, mod_bulk, mod_area:
        Elastic moduli.
    
    Returns
    -------
    float
        elastic energy
    
    """
    shape_tensor = get_shape_tensor(poly)
    area = tns.polygon_area(poly)
    delta = shape_tensor-rest_shape
    return mod_shear*(delta**2).sum()+mod_bulk*np.trace(delta)**2+mod_area*(area-A0)**2


def get_vertex_energy(poly: NDArray[Shape["*,2"],Float], A0=jnp.sqrt(3)/2, P0=1, mod_area=1, mod_perimeter=0):
    """
    Get vertex-model style energy for (ordered) polygon.
    
    E = mod_area * (area-A0)^2 + mod_perimeter * (perimeter-P0)^2
    
    """
    return mod_bulk*(tns.polygon_area(poly)-A0)**2 + mod_perimeter*(tns.polygon_perimeter(poly)-P0)**2


@patch
def get_shape_energies(self: msh.HalfEdgeMesh, mod_shear=.5, mod_bulk=1, mod_area=0, A0=np.sqrt(3)/2):
    res_dict = {}
    for v in self.vertices.values():
        neighbors = v.get_face_neighbors()
        if (None in neighbors) or self is None:
            res_dict[v._vid] = None
        else:
            cell = np.stack([fc.dual_coords for fc in neighbors])
            res_dict[v._vid] = get_shape_energy(cell, rest_shape=v.rest_shape, A0=A0,
                                                mod_shear=mod_shear, mod_bulk=mod_bulk, mod_area=mod_area)
    return res_dict

# %% ../03_real_shape_optimization.ipynb 17
@patch
def dual_vertices_to_initial_cond(self: msh.HalfEdgeMesh) -> NDArray[Shape["*"],Float]:
    """
    Format dual vertices for use in energy minimization.
    
    Returns vector of dual vertex positions (=cell vertices, associated with triangles).
    1st n_faces/2 entries are x-, 2nd n_faces/2 entries are y-coordinates.
    """
    face_keys = sorted(self.faces.keys())
    dual_vertex_vector = np.stack([self.faces[key].dual_coords for key in face_keys]).T
    return np.hstack([dual_vertex_vector[0], dual_vertex_vector[1]])
       
@patch
def initial_cond_to_dual_vertices(self: msh.HalfEdgeMesh, x0: NDArray[Shape["*"],Float]
                                 ) -> Dict[int, NDArray[Shape["2"],Float]]:
    """
    Reverse of dual_vertices_to_initial_cond, deserialize result of energy minimization.
    
    Returns dict _fcid: dual vertex position.
    """
    face_keys = sorted(self.faces.keys())
    x, y = (x0[:int(len(x0)/2)], x0[int(len(x0)/2):])
    dual_vertex_vector = np.stack([x, y], axis=1)
    return {key: val for key, val in zip(face_keys, dual_vertex_vector)}


# %% ../03_real_shape_optimization.ipynb 20
@patch
def get_primal_energy_fct_jax(self: msh.HalfEdgeMesh, bdry_list=None):
    """
    Get arrays to compute primal energy from primal vertices in JAX-compatible way.
    
    This function serializes a HalfEdgeMesh into a bunch of arrays which are used to
    compute the shape-tensor-based cell elastic energy. Boundary conditions, implemented
    as soft constraints, are passed as a list of pairs (penalty function, vertex ids),
    where 'vertex ids' is the list of vertices on a boundary, and 'penalty function'
    is the constraint potential.
        
    The function also returns an cell_list_vids array which is used internally to allow to
    make elastic moduli in the energy function cell-identity (=vertex id) dependent.
    
    Parameters
    ----------
    bdry_list: [(penalty function, vertex ids),]
        List of boundaries. None = no boundaries.
        
    Returns
    -------
    e_lst_primal: (n_edges, 2) array of ints
        Indices defining primal edges (i.e. edges of the cell tesselation)
    e_dual: (n_edges, 2) array of floats
        Unit normal vectors of dual edges corresponding to the primal edges in e_lst_primal.
        Used to enforce angle constraint
    cell_list: (n_cells, n_valence) array of ints
        Indices defining primal edges. n_valence is the number of vertices of the highest-valence
        cell in the mesh (e.g. 6 if there are only hexagons). Cells with fewer than n_valence
        vertices are padded by repeating the last vertex, so that the elastic energy is not changed.
    rest_shapes: (n_cells, 2, 2) array:
        Reference shapes of all cells as single array.
    bdry_list: list
        Slightly reformated version of input argument bdry_list.
    valence_mask: (n_cells, n_valence) array of 0/1
        Mask indicating whether there is any padding in a cell. Required internally to make the
        padding hack work.

    cell_list_vids: (n_cells) array of ints
        Array of the vertex ids corresponding to the cells in cell_list. This argument is used to
        allow elastic moduli to be a function of cell identity.
        
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

# %% ../03_real_shape_optimization.ipynb 22
@jit
def polygon_area(pts: NDArray[Shape["*,2,..."], Float]) -> NDArray[Shape["2,..."], Float]:
    """JAX-compatible - area of polygon. Assuming no self-intersection. Pts.shape (n_vertices, 2)"""
    return jnp.sum(pts[:,0]*jnp.roll(pts[:,1], 1, axis=0) - jnp.roll(pts[:,0], 1, axis=0)*pts[:,1], axis=0)/2

@jit
def get_E(x0, e_lst_primal, e_dual, cell_list, rest_shapes, bdry_list, valence_mask,
          mod_bulk=1, mod_shear=.5, angle_penalty=1000, bdry_penalty=1000, epsilon_l=1e-3,
          A0=jnp.sqrt(3)/2, mod_area=0):
    """
    Compute shape-tensor based cell elastic energy with angle & boundary constraint penalties.
    
    For mathematical details about the energy function see paper.
    
    This function relies on the arrays produced by the mesh serialization routine
    get_primal_energy_fct_jax. The first argument is the vector representing
    the primal vertex coordinates, as given by msh.HalfEdgeMesh.dual_vertices_to_initial_cond
    The other required arguments are the serialization arrays. Usage example:
    
    x0 = mesh.dual_vertices_to_initial_cond()
    energy_arrays, cell_ids = mesh.get_primal_energy_fct_jax()
    E = get_E(x0, *energy_arrays, mod_bulk=1)
    
    Parameters, i.e. the pre-factors of the different terms in the elastic energy are
    given by the keyword arguments.
    
    Parameters
    ----------
    x0 : (2*n_cell_vertices) array
        As produced by msh.HalfEdgeMesh.dual_vertices_to_initial_cond
    .... : arrays
        Serialization arrays, see msh.HalfEdgeMesh.get_primal_energy_fct_jax
    mod_bulk: float
        Shape tensor bulk modulus
    mod_shear: float
        Shape tensor shear modulus
    angle_penalty, bdry_penalty: float
        Penalties enforcing area and boundary constraints
    epsilon_l: float
        Regularization for short-length edges, required for differentiability
    A0, mod_area: float
        Reference area and area elastic modulus, can be added to shape tensor energy.
    
    Returns
    -------
    float
        Elastic energy + angle & boundary condition penalties
    
    """
    
    n_faces = int(x0.shape[0]/2)
    x, y = (x0[:n_faces], x0[n_faces:])
    pts = jnp.stack([x, y], axis=-1)
    
    # face-based shape energy
    cells = jnp.stack([pts[i] for i in cell_list.T], axis=0)
    edges = cells - jnp.roll(cells, 1, axis=0)
    lengths = jnp.sqrt(jnp.sum(edges**2, axis=-1)+epsilon_l**2)
    # + epsilon**2 to avoid non-differentiable sqrt at 0-length edges (occurs due to padding)
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
        
    # area elasticity
    areas = polygon_area(cells.transpose((0,2,1)))
    E_area = jnp.mean(mod_area*(areas-A0)**2)
    
    return E_angle + E_bdry + E_area + E_shape

get_E_jac = jit(jgrad(get_E))

# %% ../03_real_shape_optimization.ipynb 36
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
        is_bdr = np.array([any([fc.is_bdry() for fc in self.vertices[v].get_face_neighbors()])
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

# %% ../03_real_shape_optimization.ipynb 38
# for re-setting primal vertex positions after intercalation
def rotate_about_center(x, angle=pi/2):
    """Rotate pts about center. x.shape = (n_pts, 2)"""
    center = np.mean(x, axis=0)
    return (x-center)@rot_mat(angle)+np.mean(x, axis=0)

# %% ../03_real_shape_optimization.ipynb 39
def get_flip_edge(msh, minimal_l, exclude):
    primal_lengths = msh.get_primal_edge_lens(oriented=True)
    primal_lengths = sorted(primal_lengths.items(), key=lambda x: x[1])
    primal_lengths = [x for x in primal_lengths if (x[1] <= minimal_l)]
    primal_lengths = [x for x in primal_lengths if not x[0] in exclude]
    if primal_lengths:
        return primal_lengths[0][0]
    return None
