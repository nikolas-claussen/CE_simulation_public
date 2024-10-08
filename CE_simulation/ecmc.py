# AUTOGENERATED! DO NOT EDIT! File to edit: ../09b_disorder_phase_diagram.ipynb.

# %% auto 0
__all__ = ['get_valences', 'get_box', 'correct_periodic_position', 'create_crystal', 'find_all_events', 'run_ecmc',
           'remove_box_boundary', 'remove_dangling_triangles', 'create_hard_disk_initial', 'get_left_right_pt',
           'get_top_bottom_bdry', 'get_centerline', 'get_arclen', 'get_width', 'get_width_centerline',
           'get_tissue_extension']

# %% ../09b_disorder_phase_diagram.ipynb 2
import CE_simulation.mesh as msh
import CE_simulation.tension as tns
import CE_simulation.delaunay as dln
import CE_simulation.isogonal as iso
import CE_simulation.drosophila as drs
import CE_simulation.disorder as dis
import CE_simulation.hessian as hes

# %% ../09b_disorder_phase_diagram.ipynb 3
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from scipy import spatial, ndimage

from collections import Counter

from copy import deepcopy
import os
import pickle

# %% ../09b_disorder_phase_diagram.ipynb 4
import jax.numpy as jnp
from jax import grad as jgrad
from jax import jit
from jax.nn import relu

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)

# %% ../09b_disorder_phase_diagram.ipynb 5
import math
import random

# %% ../09b_disorder_phase_diagram.ipynb 6
from typing import Sequence, Union, Dict, List, Tuple, Iterable, Callable, Any
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../09b_disorder_phase_diagram.ipynb 7
import ipywidgets as widgets
import functools
from matplotlib import animation, rc

# %% ../09b_disorder_phase_diagram.ipynb 11
def get_valences(sample,keys=(4, 5, 6, 7, 8)):
    """
    Get proportion of n-sided cells (as specified by keys) from a sample.
    
    sample can be a half-edge mesh or a point cloud, in which case we use the Delaunay triangulation.
    """
    if not isinstance(sample, msh.HalfEdgeMesh):
        tri = spatial.Delaunay(sample)
        mesh = msh.HalfEdgeMesh(msh.ListOfVerticesAndFaces(tri.points, tri.simplices))
        mesh.remove_obtuse_bdry()
    else:
        mesh = sample
    valence = Counter([len(v.incident) for v in mesh.vertices.values() if not v.is_bdry()])
    valence = {key: round(valence[key]/sum(valence.values()), ndigits=3) for key in keys}
    return valence

# %% ../09b_disorder_phase_diagram.ipynb 12
def get_box(n_x, n_y, eta, shape: str):
    """Get the bounding box for ECMC that matches the initial point configuration"""
    n = n_x * n_y
    sigma = np.sqrt(eta / (n * np.pi))
    if shape == "square":
        aspect_ratio = 1.0
        box = np.array([1.0 / np.sqrt(aspect_ratio), np.sqrt(aspect_ratio)])
    elif shape == "rectangle":
        aspect_ratio = np.sqrt(3.0) / 2.0
        box = np.array([1.0 / np.sqrt(aspect_ratio), np.sqrt(aspect_ratio)])
    else:
        assert shape == "crystal"
        aspect_ratio = np.sqrt(3.0) / 2.0 * n_y / n_x
        box = np.array([1.0 / np.sqrt(aspect_ratio), np.sqrt(aspect_ratio)])    
    return box


def correct_periodic_position(position: Sequence[float], box: Sequence[float]) -> List[float]:
    """
    Return the given position corrected for periodic boundary conditions in the given simulation box.

    Parameters
    ----------
    position : Sequence[float]
        The position vector.
    box : Sequence[float]
        The geometry of the simulation box.

    Returns
    -------
    List[float]
        The position vector after considering periodic boundary condition.
    """
    return [p % b for p, b in zip(position, box)]


def create_crystal(n_x: int, n_y: int, eta: float) -> List[List[float]]:
    """
    Create an initial crystalline hard-disk configuration in the given simulation box so that the disks
    are located on the triangular lattice of a fully packed configuration.

    Parameters
    ----------
    n_x : int
        The number of disks per row in the lattice.
    n_y : int
        The number of rows in the lattice.
    eta : float
        Packing fraction.
    box : Sequence[float]
        The geometry of the box.

    Returns
    -------
    List[List[float]]
        The list of the initial two-dimensional hard-disk positions.

    Raises
    ------
    RuntimeError
        If the n_x * n_y hard disks of radius sigma do not fit in the specified simulation box.
    """
    n = n_x * n_y
    sigma = np.sqrt(eta / (n * np.pi))
    box =  get_box(n_x, n_y, eta,shape="crystal")
    n = n_x * n_y
    pos = [[0.0, 0.0] for _ in range(n)]
    distance_x = box[0] / n_x
    if distance_x < 2 * sigma:
        raise RuntimeError("The specified number of hard disks do not fit into the given simulation box.")
    distance_y = box[1] / n_y
    for i in range(n_y):
        for j in range(n_x):
            pos[i * n_x + j] = correct_periodic_position(
                [distance_x * j + 0.5 * distance_x * (i % 2), i * distance_y], box)
    pos = np.array(pos)
    return pos

# %% ../09b_disorder_phase_diagram.ipynb 15
def find_all_events(pos, pos_active, direction, sigma, box):
    """
    Compute the times when the active hard disk with a unit velocity in the given direction collides 
    with any other target disk. Also, return the distance between the two disks at the collision.

    This function is vectorized and computes the collision times and distances for all disks at once.
    The collision time is nan if the disks do not collide (the distance is ill-defined in that case).
    
    The collision time of the active disk is a special case as well and might need to be overwritten as well. 

    Parameters
    ----------
    pos : Array of shape (n_disks, 2)
        The positions of the hard disks.
    pos_active : int
        The index of the active position.
    direction : int
        The direction of the unit velocity of the active disk (0 and 1 correspond to a velocities parallel
        to  the x- and y-axes, respectively).
    sigma : float
        The radius of the hard disks.
    box : Sequence[float]
        The geometry of the box.

    Returns
    -------
    time_of_flight: np.array of shape (n_disks,)
        The time of the collision of the disks
    delta_x: np.array of shape (n_disks,)
        the distance of the two disks at the collision.)
    """    
    distance_perp = np.abs(pos[:, 1-direction] - pos[pos_active, 1-direction])
    distance_perp = np.min([distance_perp, box[1-direction] - distance_perp], axis=0)

    distance_para = pos[:, direction]-pos[pos_active, direction]
    distance_para = distance_para + (distance_para<0)*box[direction]
    
    delta_x = np.sqrt(4*sigma**2-distance_perp**2)
    time_of_flight = distance_para - delta_x
    
    # add special cases - can otherwise get stuck in 2-cycles
    time_of_flight[distance_perp >= (2*sigma)] = np.nan
    time_of_flight[distance_para == 0] = np.nan

    #np.nan_to_num(time_of_flight, nan=np.inf, copy=False)
    
    return time_of_flight, delta_x

# %% ../09b_disorder_phase_diagram.ipynb 16
def run_ecmc(initial_positions, box, eta, n_chains=100, chain_time=100, progress_bar=True):
    """
    Run ECMC mixing on a sample.
    
    The distribution of hard disks is controlled by the packing fraction eta. eta > 7.2 is the solid phase,
    eta < 7.2 is the liquid phase. To get intermediate values of hexagonal order, values between .7 and .55
    are good. 
    
    The duration of the MC mixing is controlled with the n_chains and chain_time parameters (higher=longer).
    
    
    Parameters
    ----------
    initial_positions : np.array of shape (n, 2)
        Initial positions of disks. Must be a valid configuration (I think).
    eta : float
        Packing fraction
    n_chains : int
        Number of event chains
    chain_time : float
        Time for each chain
        
    Returns
    -------
    positions : np.array of shape (n, 2)
        Sample after MCMC mixing
        
    """
    n = initial_positions.shape[0]
    sigma = np.sqrt(eta / (n * np.pi))
    positions = np.copy(initial_positions)
    
    if progress_bar:
        cbar = tqdm(range(n_chains))
        cbar.set_description("Running ECMC to create mesh")
    else: 
        cbar = range(n_chains) 
    for sample in cbar: #(cbar := tqdm(range(n_chains))):
        direction = random.randint(0, 1)
        active = random.randint(0, n - 1)
        current_chain_time = chain_time
        while current_chain_time > 0.0:
            #print(current_chain_time)
            time_of_flight, delta_x = find_all_events(positions, active, direction, sigma, box)
            time_of_flight[active] = current_chain_time
            target = np.nanargmin(time_of_flight)
            event_time, delta_x = (time_of_flight[target], delta_x[target])
            # The event time could be slightly negative due to the rounding error of the trigonometry calculation.
            # If the event time is negative, it is set to 0.0 in order to prevent the active disk moving backwards.
            positions[active, direction] += max(event_time, 0.0)
            if positions[active, direction] > box[direction]:
                positions[active, direction] -= box[direction]
            active = target
            current_chain_time -= event_time
    positions = positions % box
    
    return positions

# %% ../09b_disorder_phase_diagram.ipynb 21
def remove_box_boundary(mesh, dx=1, method="faces"):
    """
    Remove the triangles farther away then dx from the bounding box.
    
    Note: the returned mesh does not have any non-required attributes set, e.g. primal vertex positions.
    Method="faces", "vertices"
    """
    if method == "faces":
        coords = np.stack([x.coords.mean(axis=0) for x in mesh.faces.values()])
    elif method == "vertices":
        coords = np.stack([x.coords for x in mesh.vertices.values()])
    max_x, min_x = (coords[:,0].max(), coords[:,0].min())
    max_y, min_y = (coords[:,1].max(), coords[:,1].min())

    if method == "faces":
        boundary_faces = [key for key, val in mesh.faces.items() if
                          ((val.coords.mean(axis=0)[0] < (min_x+dx)) or (val.coords.mean(axis=0)[0] > (max_x-dx)) or
                           (val.coords.mean(axis=0)[1] < (min_y+dx)) or (val.coords.mean(axis=0)[1] > (max_y-dx)))]
    elif method == "vertices":
        boundary_vertices = [key for key, val in mesh.vertices.items() if
                             ((val.coords[0] < (min_x+dx)) or (val.coords[0] > (max_x-dx)) or
                              (val.coords[1] < (min_y+dx)) or (val.coords[1] > (max_y-dx)))]

    mesh_cleaned = deepcopy(mesh)
    mesh_cleaned = mesh_cleaned.to_ListOfVerticesAndFaces()
    if method == "faces":
        [mesh_cleaned.remove_face(x) for x in boundary_faces]
    elif method == "vertices":
        [mesh_cleaned.remove_vertex(x) for x in boundary_vertices]

    # remove potential orphan vertices
    mesh_cleaned = iso.CellHalfEdgeMesh(mesh_cleaned)
    bad_vertices = [key for key, val in mesh_cleaned.vertices.items() if len(val.incident) < 2]

    mesh_cleaned = mesh_cleaned.to_ListOfVerticesAndFaces()
    [mesh_cleaned.remove_vertex(x) for x in bad_vertices]
    mesh_cleaned = iso.CellHalfEdgeMesh(mesh_cleaned)

    return mesh_cleaned

# %% ../09b_disorder_phase_diagram.ipynb 22
def remove_dangling_triangles(mesh):
    """
    Remove dangling triangles connected to the mesh purely by a single vertex.
    
    Note: the returned mesh does not have any non-required attributes set, e.g. primal vertex positions.
    """
    mesh_cleaned = deepcopy(mesh)
    mesh_cleaned = mesh_cleaned.to_ListOfVerticesAndFaces()
    dangling_faces = [fc for fc in mesh.faces.values() if sum([(he.twin.face is None) for he in fc.hes]) > 2]
    [mesh_cleaned.remove_face(x._fid) for x in dangling_faces]

    return iso.CellHalfEdgeMesh(mesh_cleaned)

# %% ../09b_disorder_phase_diagram.ipynb 23
def create_hard_disk_initial(n_x, n_y, eta, initial_strain, orientation="cable",
                             remove_boundary_dx=1, intercalate=False,
                             noise_gaussian=0, isogonal=0,
                             n_chains=100, chain_time=100, progress_bar=True):
    """
    Create half-edge mesh initial condition with all necessary book-keeping from hard disk sampling.
    
    If you chose the mixing parameters (n_chains, chain_time) too low, degenerate Delaunay triangles
    can appear which mess up the data structure. So don't do that!
    
    Removes a strip of width "remove_boundary_dx" from the box edges to get rid of 
    very degenerate Delaunay triangles at the edge.
    
    noise_gaussian adds Gaussian noise on top to the triangle vertex positions.
    
    Optional: intercalate all edges with negative length. this reduces the tension anisotropy!
    
    """
    # create initial condition for ECMC
    box = get_box(n_x, n_y, eta, shape="crystal")
    initial_positions = create_crystal(n_x, n_y, eta)
    if orientation == "cable":
        box = box[::-1]
        initial_positions = initial_positions @ dln.rot_mat(np.pi/2)
        initial_positions -= initial_positions.min(axis=0)
    # run ECMC
    final_positions = run_ecmc(initial_positions, box, eta, n_chains=n_chains, chain_time=chain_time,
                               progress_bar=progress_bar)
    final_positions -= final_positions.mean(axis=0)

    # stupid step - need to get mesh to compute rescaling factor
    temp_tri = spatial.Delaunay(final_positions)
    temp_mesh = msh.HalfEdgeMesh(msh.ListOfVerticesAndFaces(temp_tri.points, temp_tri.simplices))
    rescale = 1/np.mean(list(temp_mesh.get_edge_lens().values()))
    final_positions *= rescale
    
    # add gaussian noise - need to do that _before_ computing delaunay or may generate invalid triangulation!
    if noise_gaussian > 0:
        final_positions += np.random.normal(size=final_positions.shape, scale=noise_gaussian)
    
    # convert to mesh and apply initial strain
    tri = spatial.Delaunay(final_positions)
    mesh = msh.HalfEdgeMesh(msh.ListOfVerticesAndFaces(tri.points, tri.simplices))
    mesh.transform_vertices(dln.shear_mat(1+initial_strain))
    
    # clean the boundary
    mesh = remove_box_boundary(mesh, dx=remove_boundary_dx, method="faces")
    mesh = remove_dangling_triangles(mesh)
    
    # transform to cell mesh and set initial edge lengths and primal coords
    rescale = 1/np.mean(list(mesh.get_edge_lens().values()))
    mesh.transform_vertices(lambda x: rescale*x)
    mesh.set_rest_lengths()
    mesh.set_voronoi()
    
    # intercalate negative edge lengths if desired.
    if intercalate:
        _, _ = mesh.intercalate(exclude=[], minimal_l=0, reoptimize=False)
        mesh.set_voronoi()
        for he in mesh.hes.values():
            he.passive = 0

    # set active and passive faces
    passive_faces = sorted([fc._fid for fc in mesh.faces.values() if fc.is_bdry()])
    passive_edges = list(msh.flatten([[he._heid for he in mesh.faces[fc].hes] for fc in passive_faces]))
    passive_cells = [v._vid for v in mesh.vertices.values()
                     if not v.is_bdry() and any([fc._fid in passive_faces for fc in v.faces])]

    property_dict = {"passive_faces": passive_faces, "passive_edges": passive_edges, "passive_cells": passive_cells,
                     "bdry_x": np.max([v.coords[0] for v in mesh.vertices.values()])+1,
                     "bdry_y": np.max([v.coords[1] for v in mesh.vertices.values()])+1}

    # set rest shape. isogonal mode is isotropic by default.
    for v in mesh.vertices.values():
        v.rest_shape = np.sqrt(3) * np.array([[1-isogonal, 0],[0, 1+isogonal]])

    # set area
    mean_shape = np.mean([v.get_shape_tensor() for v in mesh.vertices.values() if not v.is_bdry()], axis=0)
    scale = (2*np.sqrt(3)) / np.trace(mean_shape)
    mesh.transform_primal_vertices(scale*np.eye(2))

    return mesh, property_dict

# %% ../09b_disorder_phase_diagram.ipynb 40
@patch
def get_hexatic_order(self: msh.Vertex, use_tension_vertices=True):
    """
    Get hexatic order parameter, 1/n sum_i exp(i 6 theta_i), as complex number.
    Based on either tension vertex positions or real-space centroids.
    """
    if use_tension_vertices:
        edges = np.stack([he.vertices[0].coords-he.vertices[1].coords for he in self.incident])
    else:
        edges = np.stack([he.vertices[0].get_centroid()-he.vertices[1].get_centroid()
                          for he in self.incident])
    angles = np.stack([tns.vectors_angle(a, b) for a,b in zip(edges, np.roll(edges, 1, axis=0))])
    return np.exp(1j*6*angles).mean()

# %% ../09b_disorder_phase_diagram.ipynb 66
from skimage import graph

def get_left_right_pt(mesh):
    """Use initial mesh to get starting points for centerline construction"""
    #coords = np.stack([v.get_centroid() for v in mesh.vertices.values()])
    coords = np.stack([v.coords for v in mesh.vertices.values()]) # use triangle coords

    max_x, min_x = (coords[:,0].max(), coords[:,0].min())
    med_y = np.median(coords[:,1])

    min_key, _ = min(mesh.vertices.items(), key=lambda x: np.linalg.norm(x[1].coords-np.array([min_x, med_y])))
    max_key, _ = min(mesh.vertices.items(), key=lambda x: np.linalg.norm(x[1].coords-np.array([max_x, med_y])))
    
    return min_key, max_key


def get_top_bottom_bdry(mesh, tol=1):
    """Use initial mesh to get top/bottom boundary to measure height with centerline"""
    #coords = np.stack([v.get_centroid() for v in mesh.vertices.values()])
    coords = np.stack([v.coords for v in mesh.vertices.values()]) # use triangle coords
    
    max_y, min_y = (coords[:,1].max(), coords[:,1].min())
    
    top_bdry = [v._vid for v in mesh.vertices.values() if v.is_bdry() and np.abs(v.coords[1]-max_y) < tol]
    bottom_bdry = [v._vid for v in mesh.vertices.values() if v.is_bdry() and np.abs(v.coords[1]-min_y) < tol]

    return top_bdry, bottom_bdry


def get_centerline(mesh, start, stop, dx=.5, sigma=2):
    """Use Noah's fast-marching method to get a centerline"""
    
    # create a coordinate grid array.
    coords = np.stack([v.get_centroid() for v in mesh.vertices.values()])

    xlims = (coords[:,0].min()-2*dx, coords[:,0].max()+2*dx)
    ylims = (coords[:,1].min()-2*dx, coords[:,1].max()+2*dx)

    x = np.linspace(xlims[0], xlims[1], round((xlims[1]-xlims[0])/dx))
    y = np.linspace(ylims[0], ylims[1], round((ylims[1]-ylims[0])/dx))
    X, Y = np.meshgrid(x,y)
    r = np.stack([X,Y], axis=-1)

    # now create a binary mask indicating the location of the mesh
    # by setting pixel at vertex positiions to one and dilating.
    
    mask = np.zeros_like(X)
    for p in coords:
        dist = np.linalg.norm(r-p, axis=-1)
        ind = np.unravel_index(np.argmin(dist), dist.shape)
        mask[-(ind[0]+1), ind[1]] = 1
    mask = ndimage.binary_dilation(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)

    # compute the distance transform
    distance_tf = ndimage.distance_transform_edt(mask)
    distance_tf = distance_tf.max()-distance_tf
    distance_tf_neg = ndimage.distance_transform_edt(1-mask)
    distance_tf = distance_tf + distance_tf_neg    
    
    # now use fastest path finding
    start = np.unravel_index(np.argmin(np.linalg.norm(r-mesh.vertices[start].get_centroid(), axis=-1)), X.shape)
    end = np.unravel_index(np.argmin(np.linalg.norm(r-mesh.vertices[stop].get_centroid(), axis=-1)), X.shape)
    
    path, length = graph.route_through_array(distance_tf**2, (-start[0], start[1]), (-end[0], end[1]))
    path = np.stack(path)
    path_geom = np.stack([X[path[:,0], path[:,1]], Y[-path[:,0], path[:,1]]], axis=1)
    path_geom = ndimage.gaussian_filter1d(path_geom, sigma=sigma, axis=0) #[sigma:-sigma]
    
    return path_geom

def get_arclen(path):
    """compute arc length of path (n, dim) array"""
    return np.sum(np.linalg.norm(path[1:]-path[:-1], axis=1), axis=0)


def get_width(mesh, top_bdry, bottom_bdry):
    """No need a priori for the center line"""
    bottom_coords = np.stack([mesh.vertices[v].get_centroid() for v in bottom_bdry])
    top_bottom_dist = np.array([np.linalg.norm(bottom_coords-mesh.vertices[v].get_centroid(), axis=-1).min()
                                for v in top_bdry])
    return np.median(top_bottom_dist)

def get_width_centerline(mesh, top_bdry, bottom_bdry, centerline):
    """No need a priori for the center line"""
    top_dist = [np.linalg.norm(path_geom-mesh.vertices[v].get_centroid(), axis=1).min() for v in top_bdry]
    bottom_dist = [np.linalg.norm(path_geom-mesh.vertices[v].get_centroid(), axis=1).min() for v in bottom_bdry]

    return 2*np.median(top_dist+bottom_dist)

# %% ../09b_disorder_phase_diagram.ipynb 94
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
