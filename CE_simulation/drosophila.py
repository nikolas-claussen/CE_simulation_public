# AUTOGENERATED! DO NOT EDIT! File to edit: ../04_drosophila_simulation.ipynb.

# %% auto 0
__all__ = ['fridtjof_colors', 'create_rect_initial', 'plot_mesh', 'get_p_over_sqrt_A']

# %% ../04_drosophila_simulation.ipynb 3
import CE_simulation.mesh as msh
import CE_simulation.tension as tns
import CE_simulation.delaunay as dln
import CE_simulation.isogonal as iso

# %% ../04_drosophila_simulation.ipynb 4
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import optimize, ndimage

from tqdm.notebook import tqdm

from copy import deepcopy
import pickle

# %% ../04_drosophila_simulation.ipynb 5
from typing import Union, Dict, List, Tuple, Iterable, Callable, Any
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../04_drosophila_simulation.ipynb 6
import jax.numpy as jnp
from jax import grad as jgrad
from jax import jit
from jax.tree_util import Partial

from jax.config import config
config.update("jax_enable_x64", True) # 32 bit leads the optimizer to complain about precision loss
config.update("jax_debug_nans", False) # useful for debugging, but makes code slower!

# %% ../04_drosophila_simulation.ipynb 7
import ipywidgets as widgets
import functools
from matplotlib import animation, rc

# %% ../04_drosophila_simulation.ipynb 11
## colors for cell rows

fridtjof_colors = np.array([[0.34398, 0.49112, 0.89936],
                            [0.97, 0.606, 0.081],
                            [0.91, 0.318, 0.243],
                            [0.448, 0.69232, 0.1538],
                            [0.62168, 0.2798, 0.6914],
                            [0.09096, 0.6296, 0.85532],
                            [0.46056, 0.40064, 0.81392],
                            [0.94, 0.462, 0.162],
                            [0., 0.7, 0.7],
                            [0.827051, 0.418034, 0.0243459],
                            [0.5511749434976025, 0.32014794962639853, 0.8720626412559938],
                            [0.72694101250947, 0.7196601125010522, 0.],
                            [0.8680706456216862, 0.2563858708756628, 0.30321559063052295],
                            [0.2418693812442152, 0.5065044950046278, 0.9902432574930582],
                            [0.9573908706237908, 0.5369543531189542, 0.11504464931576472]])

# %% ../04_drosophila_simulation.ipynb 12
def create_rect_initial(nx, ny, noise=0, initial_strain=0, isogonal=0, orientation='orthogonal',
                        boundaries=None, w_passive=0, bdry_x=None, bdry_y=None,
                        random_seed=0):
    """
    Create initial condition for germ band simulations.
    
    Creates a CellHalfEdgeMesh from a rectangulat patch of triangular lattice, creates boundary conditions,
    selects active and passive triangles, creates a dictionary with the initial rows of active cells,
    and sets edge rest lengths and cell rest shapes.
    
    Random seed for noise can be specified for reproducible results.
    
    Parameters
    ----------
    nx, ny: int
        Width and height of rectangular patch
    noise: float
        Standard deviation of the noise added to the initial vertex positions
    initial_strain: float
        Initial y-axis strain applied to tension triangulation. Applies transform matrix
        diag(1/(1+s), 1+s)
    orientation: 'orthogonal' or 'parallel'
        Orientation of hexagonal lattice direction w.r.t. y-axis.
    boundaries: list of 'top', 'bottom', 'left', 'right'
        On which sides to add slip walls.
    w_passive: float
        Width of passive region on the top and bottom
    bdry_x, bdry_y: float
        Location of the the left/right and top/bottom slip walls. If None, set to initial mesh positions
    random_seed: int
        Numpy random number generator seed.
        
    """
    np.random.seed(random_seed)
    # create the mesh
    if orientation == 'parallel':
        nx, ny = (ny, nx)
    mesh_initial = iso.CellHalfEdgeMesh(dln.create_rect_mesh(ny, nx, noise=noise, defects=(0, 0),
                                                             straight_bdry=False))
    if orientation == 'orthogonal':
        mesh_initial.transform_vertices(dln.rot_mat(np.pi/2))
    center = np.mean([v.coords for v in mesh_initial.vertices.values()], axis=0)
    mesh_initial.transform_vertices(lambda x: x-center)
    mesh_initial.set_voronoi()
    
    mesh_initial.transform_vertices(dln.shear_mat(1+initial_strain))
    mesh_initial.set_rest_lengths()
    
    # create the boundary conditions
    boundaries = [] if boundaries is None else boundaries
    bdry_list = []
    max_x_cells = np.max([v.get_centroid()[0] for v in mesh_initial.vertices.values() if not v.is_bdry()])
    max_y_cells = np.max([v.get_centroid()[1] for v in mesh_initial.vertices.values() if not v.is_bdry()])

    bdry_x = np.ceil(max_x_cells) if bdry_x is None else bdry_x
    bdry_y = np.ceil(max_y_cells) if bdry_y is None else bdry_y
    
    w_bdry = .4
    
    if 'top' in boundaries:
        top_ids = []
        for v in mesh_initial.vertices.values():
            if (v.get_centroid()[1] > (max_y_cells-w_bdry)) and (not v.is_bdry()):
                top_ids.append(v._vid)
        def top_penalty(x):
            return (x[1]-bdry_y)**2
        top_penalty = Partial(jit(top_penalty))
        bdry_list.append([top_penalty, top_ids])

    if 'bottom' in boundaries:
        bottom_ids = []
        for v in mesh_initial.vertices.values():
            if (v.get_centroid()[1] < -(max_y_cells-w_bdry)) and (not v.is_bdry()):
                bottom_ids.append(v._vid)
        def bottom_penalty(x):
            return (x[1]+bdry_y)**2
        bottom_penalty = Partial(jit(bottom_penalty))
        bdry_list.append([bottom_penalty, bottom_ids])
        
    if 'left' in boundaries:
        left_ids = []
        for v in mesh_initial.vertices.values():
            if (v.get_centroid()[0] < -(max_x_cells-w_bdry)) and (not v.is_bdry()):
                left_ids.append(v._vid)
        def left_penalty(x):
            return (x[0]+bdry_x)**2
        left_penalty = Partial(jit(left_penalty))
        bdry_list.append([left_penalty, left_ids])

    if 'right' in boundaries:
        right_ids = []
        for v in mesh_initial.vertices.values():
            if (v.get_centroid()[0] > (max_x_cells-w_bdry)) and (not v.is_bdry()):
                right_ids.append(v._vid)
        def right_penalty(x):
            return (x[0]-bdry_x)**2
        right_penalty = Partial(jit(right_penalty))
        bdry_list.append([right_penalty, right_ids])

    # set the active and passive triangles
    passive_faces = []
    max_y_faces = np.max([val.dual_coords[1] for val in mesh_initial.faces.values()])
    for fc in mesh_initial.faces.values():
        if fc.is_bdry() or (np.abs(fc.dual_coords[1]) > (max_y_faces-w_passive)):
            passive_faces.append(fc._fid)
    passive_faces = sorted(passive_faces)
    passive_edges = msh.flatten([[he._heid for he in mesh_initial.faces[fc].hes] for fc in passive_faces])
    passive_cells = [v._vid for v in mesh_initial.vertices.values()
                     if not v.is_bdry() and any([fc._fid in passive_faces for fc in v.get_face_neighbors()])]

    # create dict of initial row ids
    if orientation == 'parallel':
        initial_row_dict = {key: np.round((2/np.sqrt(3))*val.get_centroid()[1]+.5, decimals=0)
                            for key, val in mesh_initial.vertices.items()
                            if (not key in passive_cells) and (not val.is_bdry())}
    elif orientation == 'orthogonal':
        initial_row_dict = {key: np.round(val.get_centroid()[1], decimals=0)
                            for key, val in mesh_initial.vertices.items()
                            if (not key in passive_cells) and (not val.is_bdry())}
    min_val = min(initial_row_dict.values())
    initial_row_dict = {key: int(val-min_val) for key, val in initial_row_dict.items()}
    
    # set isogonal mode for active cells.
    for v in mesh_initial.vertices.values():
        if v._vid in passive_cells:
            v.rest_shape = np.sqrt(3) * np.array([[1, 0],[0, 1]])
        else:
            v.rest_shape = np.sqrt(3) * np.array([[1-isogonal, 0],[0, 1+isogonal]])

    property_dict = {'initial_row_dict': initial_row_dict, 'passive_faces': passive_faces,
                     'passive_edges': passive_edges, 'passive_cells': passive_cells,
                     'bdry_x': bdry_x, 'bdry_y': bdry_y}
    
    return mesh_initial, bdry_list, property_dict

# %% ../04_drosophila_simulation.ipynb 36
def plot_mesh(i, xlim, ylim, mesh_series, flipped_series=None,
              edge_colors=None, cell_colors=None, slipwall_y=None, plot_cell=True, plot_tri=False):
    ''"""
    Plot time series of meshes (simulation results).
    
    This function is used primarily for interactive slider plots, and to create movies.
    Usage example, using widget.fixed to set the args you do not want to tune interactively:
    
    fig = plt.figure(figsize=(6, 6))
    widgets.interact(plot_mesh, i=(0, len(meshes)-1, 1), xlim=(bdry_x, 2*bdry_x),
                 ylim=widgets.fixed(bdry_y+.5), edge_colors=None, cell_colors=None,
                 mesh_series=widgets.fixed(meshes), flipped_series=widgets.fixed(last_flipped_edges));
    
    Parameters
    ----------
    i: int
        Time point to plot
    xlim, ylim: float
        x- and y- limits of the axes, symmetric about 0.
    mesh_series, flipped_series: list
        Time series of meshes and T1 events, as given by simulation loop
    edge_colors, cell_colors: dict
        color dict, see mesh.cell_plot
    slipwall_y: float or None
        If float, plot slip walls at top/bottom at this position. If None, don't plot anything. 
    plot_cell, plot_tri: bool
        plot cells and/or triangulation
    """
    
    flipped_series = [] if flipped_series is None else flipped_series
    plt.cla()
    if slipwall_y is not None:
        plt.hlines((bdry_y, -bdry_y), (-xlim, -xlim), (xlim, xlim), color="k")
    plt.xlim([-xlim, xlim])
    plt.ylim([-ylim, ylim])
    plt.gca().set_aspect("equal", adjustable="box")
    if plot_cell:
        mesh_series[i].cellplot(edge_colors=edge_colors,
                                cell_colors=cell_colors)
    #meshes[i].labelplot(halfedge_labels=True, vertex_labels=True, face_labels=False)
    if plot_tri:
        mesh_series[i].triplot()
    plt.title(i)
    for x in flipped_series[i+1]:
        he = meshes[i].hes[x]
        if plot_cell:
            line = np.stack([he.face.dual_coords, he.twin.face.dual_coords])
            plt.plot(*line.T, c="r", lw=4)
        if plot_tri:
            line = np.stack([he.vertices[0].coords, he.vertices[1].coords])
            plt.plot(*line.T, c="tab:purple", lw=5)

# %% ../04_drosophila_simulation.ipynb 55
def get_p_over_sqrt_A(v: msh.Vertex) -> float:
    """Compute perimeter/sqrt(area) of cell. Returns None for boundary cells."""
    if v.is_bdry():
        return None
    cell = np.stack([fc.dual_coords for fc in v.get_face_neighbors()])
    return tns.polygon_perimeter(cell) / np.sqrt(tns.polygon_area(cell))
