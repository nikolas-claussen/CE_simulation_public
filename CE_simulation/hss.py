# AUTOGENERATED! DO NOT EDIT! File to edit: ../06_isogonal_hessian.ipynb.

# %% auto 0
__all__ = ['get_E_iso_jac', 'get_E_iso_hessian', 'get_E_iso', 'top_q_share']

# %% ../06_isogonal_hessian.ipynb 2
import CE_simulation.mesh as msh
import CE_simulation.tension as tns
import CE_simulation.delaunay as dln
import CE_simulation.isogonal as iso

# %% ../06_isogonal_hessian.ipynb 3
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import optimize, ndimage, sparse

from tqdm.notebook import tqdm

from copy import deepcopy
import pickle

# %% ../06_isogonal_hessian.ipynb 4
from typing import Union, Dict, List, Tuple, Iterable, Callable, Any
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../06_isogonal_hessian.ipynb 5
import jax.numpy as jnp
from jax import jit
import jax
from jax.tree_util import Partial

from jax.config import config
config.update("jax_enable_x64", True) # 32 bit leads the optimizer to complain about precision loss
config.update("jax_debug_nans", False) # useful for debugging, but makes code slower!

# %% ../06_isogonal_hessian.ipynb 6
import ipywidgets as widgets
import functools
from matplotlib import animation, rc

# %% ../06_isogonal_hessian.ipynb 8
@patch
def get_isogonal(self: msh.Vertex) -> Dict[int, NDArray[Shape["2"], Float]]:
    """
    Get isogonal mode for a given cell.
    
    Returns a dict: {faceid: translation vector} of primal vertices. Entries are
    
    j: T_j,bc / S_{self}bc
    
    where self,b,c are the three cells meeting at vertex j, and S is the area of the tension triangle.
    T_j is pointing towards vertex j (i.e. inwards)
    
    """

    # iterate over faces (cell vertices) adjacent to self. note: these are ccwise ordered by construction.
    isogonal_dict = {}
    
    nghbs = self.get_face_neighbors()
    for fc in nghbs:
        if fc is not None:
            S = fc.get_area()
            he = next(he for he in fc.hes if not self in he.vertices)            
            # note: hes are oriented ccwise
            T = dln.rot_mat(-np.pi/2)@(he.vertices[1].coords - he.vertices[0].coords)
            isogonal_dict[fc._fid] = T/S
        
    return isogonal_dict

# %% ../06_isogonal_hessian.ipynb 17
@patch
def get_isogonal_transform_matrix(self: msh.HalfEdgeMesh, flattened=False) -> NDArray[Shape["*,*,2"], Float]:
    """
    Create a matrix that transforms isogonal modes to vertex displacement
    
    To do the basis conversion, order vertices (cells) and faces (cell vertices) according to their indices.
    
    If flattened, flatten by combining face & x/y-component index
    
    """    
    face_key_dict = {key: ix for ix, key in enumerate(sorted(self.faces.keys()))}
    vertex_key_dict = {key: ix for ix, key in enumerate(sorted(self.faces.keys()))}
    
    iso_matrix = np.zeros((len(self.vertices), len(self.faces), 2))
    iso_dicts = {key: val.get_isogonal() for key, val in self.vertices.items()}

    for vkey, iso_dict in iso_dicts.items():
        for fkey, dr in iso_dict.items():
            iso_matrix[vertex_key_dict[vkey], face_key_dict[fkey], :] = dr
    if flattened:
        return iso_matrix.reshape((iso_matrix.shape[0], iso_matrix.shape[1]*iso_matrix.shape[2]))
    return iso_matrix

# %% ../06_isogonal_hessian.ipynb 27
@jit
def get_E_iso(x0, e_lst_primal, e_dual, cell_list, bdry_list, valence_mask,
              mod_bulk=0, mod_shear=0, shape0=jnp.sqrt(3)*jnp.eye(2),
              mod_area=0, A0=jnp.sqrt(3)/2, mod_perimeter=0, P0=2*jnp.sqrt(3),
              angle_penalty=0, bdry_penalty=0, epsilon_l=1e-3):
    """
    Compute shape-tensor based cell elastic energy with angle & boundary constraint penalties.
    
    For mathematical details about the energy function see paper.
    
    This function relies on the arrays produced by the mesh serialization routine
    get_primal_energy_fct_jax. The first argument is the vector representing
    the primal vertex coordinates, as given by msh.HalfEdgeMesh.primal_vertices_to_vector
    The other required arguments are the serialization arrays. Usage example:
    
    x0 = mesh.primal_vertices_to_vector()
    energy_arrays, cell_ids = mesh.get_primal_energy_fct_jax()
    E = get_E(x0, *energy_arrays, mod_bulk=1)
    
    Parameters, i.e. the pre-factors of the different terms in the elastic energy are
    given by the keyword arguments.
    
    Parameters
    ----------
    x0 : (2*n_cell_vertices) array
        As produced by msh.HalfEdgeMesh.primal_vertices_to_vector
    .... : arrays
        Serialization arrays, see msh.HalfEdgeMesh.get_primal_energy_fct_jax
    mod_bulk, mod_shear, shape0: float
        Moduli and reference shape for shape tensor elasticity
    mod_area, mod_perimetet, A0, Po: float
        Moduli and reference area/perimeter for vertex-model elasticity
    angle_penalty: float
        angle penalty strength
    bdry_penalty: float
        boundary condition strength
    epsilon_l: float
        Regularization for short-length edges, required for differentiability

    Returns
    -------
    float
        Elastic energy + angle & boundary condition penalties
    
    """
    pts = jnp.reshape(x0, (int(x0.shape[0]/2), 2))
    cells = jnp.stack([pts[i] for i in cell_list.T], axis=0)
    
    # area+perimeter elasticity
    areas = tns.polygon_area(cells.transpose((0,2,1)))
    perimeters = tns.polygon_perimeter(cells.transpose((0,2,1)))
    E_vertex = mod_area*jnp.mean((areas-A0)**2) + mod_perimeter*jnp.mean((perimeters-P0)**2)
    
    # face-based shape energy
    edges = cells - jnp.roll(cells, 1, axis=0)
    lengths = jnp.sqrt(jnp.sum(edges**2, axis=-1)+epsilon_l**2)
    # + epsilon**2 to avoid non-differentiable sqrt at 0-length edges (occurs due to padding)
    units = (edges.T/lengths.T).T
    delta = jnp.einsum('efi,efj->fij', edges, units) - shape0
    E_shape = (mod_shear*jnp.mean(jnp.sum(delta**2, axis=(1,2)))
               + mod_bulk*jnp.mean((delta[:,0,0]+delta[:,1,1])**2))
        
    # angle penalty
    e_primal = pts[e_lst_primal[:,1],:] - pts[e_lst_primal[:,0],:] # he.twin.face-he.face
    lengths = jnp.sqrt(jnp.sum(e_primal**2, axis=-1)+epsilon_l**2)
    # + epsilon to avoid 0-division error and make penalty smooth as length passes through 0
    E_angle = angle_penalty*jnp.mean(1-jnp.einsum('ei,ei->e', e_primal, e_dual)/lengths)
    # note: non-zero epsilon creates a "penalty" against 0-length edges 

    # boundary conditions
    E_bdry = 0
    for bdry in bdry_list:
        centroids = (jnp.sum(cells[:,bdry[1]].T*valence_mask[bdry[1]], axis=-1)
                     /jnp.sum(valence_mask[bdry[1]],axis=-1))
        E_bdry = E_bdry + bdry_penalty*jnp.sum(bdry[0](centroids)) # shape (2, n_cells_in_bdry)
    
    return E_vertex + E_shape + E_angle + E_bdry

get_E_iso_jac = jit(jax.grad(get_E_iso))
get_E_iso_hessian = jit(jax.hessian(get_E_iso))

# %% ../06_isogonal_hessian.ipynb 28
@patch
def get_iso_energy_fct_jax(self: iso.CellHalfEdgeMesh, bdry_list=None):
    """Get the relevant subset of serialization arrays"""
    (e_lst_primal, e_dual, cell_list, _, bdry_list, valence_mask), _ = self.get_primal_energy_fct_jax(bdry_list)
    return e_lst_primal, e_dual, cell_list, bdry_list, valence_mask


# %% ../06_isogonal_hessian.ipynb 29
@patch
def optimize_cell_shape(self: iso.CellHalfEdgeMesh, bdry_list=None,
                        energy_args=None, tol=1e-3, maxiter=10000, verbose=True) -> Dict:
    """
    Set primal vertex positions by constrained cell-shape energy optimization.
    
    The parameters for the elastic energy (e.g. moduli) are passed as a dict 'energy_args'.
    bdry_list contains the boundary conditions, as pairs (penalty function, vertex ids).
    Also passes arguments to the scipy optimizer ('tol', 'maxiter')
        
    Parameters
    ----------
    bdry_list: [(penalty function, vertex ids),]
        List of boundaries. None = no boundaries.
    energy_args: Dict
        Dictionary with the parameters to the cell shape elastic energy. See `get_E_iso`.
    tol: float
        Optimizer tolerance
    maxiter: int
        Maximum number of optimizer iterations
    verbose: bool
        Print error messages
    
    Returns
    -------
    None
    
    """
    x0 = self.primal_vertices_to_vector()
    get_E_arrays = self.get_iso_energy_fct_jax(bdry_list)
    
    # set default arguments and convert to list for optimizer
    default_args = {'mod_bulk': 0, 'mod_shear': 0, 'shape0': jnp.sqrt(3)*jnp.eye(2),
                    'mod_area': 0, 'A0': jnp.sqrt(3)/2, 'mod_perimeter': 0, 'P0': 2*jnp.sqrt(3),
                    'angle_penalty': 0, 'bdry_penalty': 0, 'epsilon_l': 1e-3}
    combined = {} if energy_args is None else deepcopy(energy_args)
    for key, val in default_args.items():
        if key not in combined:
            combined[key] = val
    args_list = (combined[key] for key in ['mod_bulk', 'mod_shear', 'shape0', 'mod_area', 'A0',
                                           'mod_perimeter', 'P0', 'angle_penalty', 'bdry_penalty', 'epsilon_l'])
    
    # optimize
    sol = optimize.minimize(get_E, x0, jac=get_E_jac, args=get_E_arrays+args_list,
                             method="CG", tol=tol, options={"maxiter": maxiter})
    
    if sol["status"] !=0 and verbose:
        print("Cell shape optimization failed", sol["message"])
    new_coord_dict = self.vector_to_primal_vertices(sol["x"])
    for key, val in self.faces.items():
        val.dual_coords = new_coord_dict[key]
    return sol

# %% ../06_isogonal_hessian.ipynb 43
def top_q_share(x, q=.9):
    return np.round(x[x > np.quantile(x, q)].sum() / x.sum(), decimals=2)
