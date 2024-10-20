# AUTOGENERATED! DO NOT EDIT! File to edit: ../09_disorder_extension_simulation.ipynb.

# %% auto 0
__all__ = ['hard_disk_radius', 'rect_mask', 'create_poisson_disk_initial', 'cable_dt_act_pass', 'get_tissue_extension']

# %% ../09_disorder_extension_simulation.ipynb 2
import CE_simulation.mesh as msh
import CE_simulation.tension as tns
import CE_simulation.delaunay as dln
import CE_simulation.isogonal as iso
import CE_simulation.drosophila as drs

# %% ../09_disorder_extension_simulation.ipynb 3
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection, PatchCollection

from scipy import optimize, ndimage

from tqdm.notebook import tqdm

from copy import deepcopy
import pickle

# %% ../09_disorder_extension_simulation.ipynb 4
from typing import Union, Dict, List, Tuple, Iterable, Callable, Any
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../09_disorder_extension_simulation.ipynb 5
import jax.numpy as jnp
from jax import grad as jgrad
from jax import jit
from jax.tree_util import Partial

from jax.config import config
config.update("jax_enable_x64", True) # 32 bit leads the optimizer to complain about precision loss
config.update("jax_debug_nans", False) # useful for debugging, but makes code slower!

# %% ../09_disorder_extension_simulation.ipynb 6
import ipywidgets as widgets
import functools
from matplotlib import animation, rc

# %% ../09_disorder_extension_simulation.ipynb 7
from collections import Counter

# %% ../09_disorder_extension_simulation.ipynb 11
# use the scipy quasi monte carlo sampler to generate samples of hard disk point process.
# then Delaunay that, and shear to get initial condition

from scipy.stats import qmc
from scipy import spatial

# %% ../09_disorder_extension_simulation.ipynb 16
# we want to auto-tune the radius to get the desired number of candidates.
# assume a random packing fraction of like 75% (https://doi.org/10.1103/PhysRevA.41.4199)
# in truth, the qmd algorithm spits out lower densities, like 60%. and we also need to remove boundary pts
# aspect ratio correction if we want to select a rectangle etc ou of the square

def hard_disk_radius(desired_samples, aspect=1, density=.55):
    return 2*np.sqrt((density/aspect)/(desired_samples*np.pi))

# %% ../09_disorder_extension_simulation.ipynb 18
@patch
def get_edge_lens(self: msh.Face):
    vs = [v.coords for v in self.vertices]
    ls = np.array([np.linalg.norm(vs[0]-vs[1]),
                   np.linalg.norm(vs[1]-vs[2]),
                   np.linalg.norm(vs[2]-vs[0])])
    return ls

@patch
def remove_obtuse_bdry(self: msh.HalfEdgeMesh, iterations=2, min_obtuse=1):
    """
    Remove all boundary triangles from mesh iteratively
    
    We often like to remove boundary triangles when creating a mesh from a Delaunay triangulation
    on some point set. The original Delaunay boundary edges will be very long. Unfortunately I have
    only implemented this for pre-meshes..., hence the wrapper.
    
    Can filter and remove only edges with very long sides (longest side min_obtuse times longer than shortest).
    """
    premesh = self.to_ListOfVerticesAndFaces()
    mesh_new = deepcopy(self)
    for i in range(0, iterations):
        remove = [key for key, fc in mesh_new.faces.items() if fc.is_bdry_edge()
                  and (np.max(fc.get_edge_lens())/np.min(fc.get_edge_lens())) > min_obtuse]
        for key in remove:
            premesh.remove_face(key)
        mesh_new = msh.HalfEdgeMesh(premesh)
        
    return mesh_new

# %% ../09_disorder_extension_simulation.ipynb 21
def rect_mask(x, aspect=1):
    """Aspect > 1 corresponds to tall rectangles"""
    min_y, max_y = (0, 1)
    dx = 1- 1/aspect
    min_x, max_x = (dx/2, 1-dx/2)
    return (min_y <= x[1] <= max_y) and (min_x <= x[0] <= max_x)

# %% ../09_disorder_extension_simulation.ipynb 27
# package this into a function


def create_poisson_disk_initial(n_vertices, aspect=1, initial_strain=0.1, isogonal=0, random_seed=1):
    """
    Create (rectangular shaped) Poisson hard disk Voronoi mesh.
    
    Sample points from a hard disk Poisson distribution using scipy quasi-monte carlo module.
    Use it to create a mesh using Delauny. There are a bunch of post-processing steps(select rectangle with
    desired aspect ratio, remove long boundary edges, set active/passive regions, shear).
    
    Can take a bit of time to sample since we want the disks closely packed and the algorithm is naive.
    
    Note: no boundaries.
    
    Parameter
    ---------
    n_vertices: int
        Approximate number of desired vertices. Not guaranteed to match exactly
    initial_strain: float
        Initial y-axis strain applied to tension triangulation. Applies transform matrix
        diag(1/(1+s), 1+s)
    isogonal : float
        isogonal mode, incorporated into reference shape tensors. 0 = isotropic, >0 y-axis elongated
    random_seed: int
        Numpy random number generator seed.
    
    Returns
    -------
    mesh: iso.CellHalfEdgeMesh
    
    property_dict: dict
        dictionary with bounding box size and active/passive cells/edges in the patch
    
    """
    # sample points
    aspect = aspect * (1+initial_strain)/(1-initial_strain) # adjust aspect for shear
    radius = hard_disk_radius(n_vertices, aspect=aspect)
    ncandidates = 4*n_vertices  # determines density. high values = slow.
    engine = qmc.PoissonDisk(d=2, radius=radius, hypersphere='volume', ncandidates=ncandidates,
                             optimization=None, seed=random_seed)
    sample = engine.random(int(n_vertices*aspect))  
    rect_sample = np.stack([x for x in sample if rect_mask(x, aspect=aspect)])

    # use Delaunay to create mesh
    tri = spatial.Delaunay(rect_sample)
    premesh = msh.ListOfVerticesAndFaces(tri.points, tri.simplices)
    mesh = msh.HalfEdgeMesh(premesh)
    # remove all the original Delaunay boundary edges which will be very long
    mesh = mesh.remove_obtuse_bdry(iterations=6, min_obtuse=2)
    
    # rescale, and shear
    rescale = 1/np.mean(list(mesh.get_edge_lens().values()))
    mesh.transform_vertices(rescale*dln.shear_mat(1+initial_strain))
    mean = np.mean([v.coords for v in mesh.vertices.values()], axis=0)
    mesh.transform_vertices(lambda x: x-mean)

    # transform to cell mesh and set initial edge lengths and primal coords
    mesh = iso.CellHalfEdgeMesh(mesh)
    mesh.set_rest_lengths()
    mesh.set_voronoi()
    
    # set active and passive faces
    passive_faces = sorted([fc._fid for fc in mesh.faces.values() if fc.is_bdry()])
    passive_edges = list(msh.flatten([[he._heid for he in mesh.faces[fc].hes] for fc in passive_faces]))
    passive_cells = [v._vid for v in mesh.vertices.values()
                     if not v.is_bdry() and any([fc._fid in passive_faces for fc in v.faces])]

    property_dict = {"passive_faces": passive_faces, "passive_edges": passive_edges, "passive_cells": passive_cells,
                     "bdry_x": np.max([v.coords[0] for v in mesh.vertices.values()])+1,
                     "bdry_y": np.max([v.coords[1] for v in mesh.vertices.values()])+1}
    
    # set isogonal mode .
    for v in mesh.vertices.values():
        if v._vid in passive_cells:
            v.rest_shape = np.sqrt(3) * np.array([[1, 0],[0, 1]])
        else:
            v.rest_shape = np.sqrt(3) * np.array([[1-isogonal, 0],[0, 1+isogonal]])

    return mesh, property_dict

# %% ../09_disorder_extension_simulation.ipynb 43
def cable_dt_act_pass(Ts, Tps, m, k=1, T_minus=0, T_c=1, T_plus=1.25):
    """
    Time derivative of tension for making cables. m used to mark passive cells (m == 1)
    """
    dT_dt = (m!=1)*(-(Ts-T_minus)*(Ts-T_c)*(Ts-T_plus)) - k*(m==1)*(Ts-1)
    dT_dt -= dT_dt.mean()
    dTp_dt = -k*Tps
    return dT_dt, dTp_dt

# %% ../09_disorder_extension_simulation.ipynb 77
def get_tissue_extension(meshes, sigma=2, q=0.9, log=True, exclude=None):
    """Get tissue extension by means of q% x- and y-axis bounding box"""
    exclude = [] if exclude is None else exclude
    centroids = np.stack([[v.get_centroid() for key, v in mesh.vertices.items()
                           if (not v.is_bdry()) and (not v._vid in exclude)
                          ] for mesh in meshes[1:]])

    delta = np.quantile(centroids, axis=1, q=q)-np.quantile(centroids, axis=1, q=1-q,)
    delta_smooth = ndimage.gaussian_filter1d(delta, axis=0, sigma=sigma)[sigma:-sigma]
    delta_smooth /= delta_smooth[0]
    if log:
        delta_log = np.log(delta_smooth)
        return delta_log
    return delta_smooth
