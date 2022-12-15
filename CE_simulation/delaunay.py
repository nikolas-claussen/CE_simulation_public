# AUTOGENERATED! DO NOT EDIT! File to edit: ../02_delaunay_simulation.ipynb.

# %% auto 0
__all__ = ['rot_mat', 'shear_mat', 'scale_mat', 'get_triangular_lattice_convex', 'get_tri_hemesh', 'get_triangular_lattice',
           'create_rect_mesh', 'create_rect_mesh_angle', 'get_inertia', 'get_conformal_transform']

# %% ../02_delaunay_simulation.ipynb 3
import CE_simulation.mesh as msh
import CE_simulation.tension as tns

# %% ../02_delaunay_simulation.ipynb 4
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import trim_mean
from scipy.linalg import orthogonal_procrustes
from scipy import spatial

from tqdm.notebook import tqdm

from math import floor, ceil

from copy import deepcopy

# %% ../02_delaunay_simulation.ipynb 5
import jax.numpy as jnp
from jax import grad as jgrad
from jax import jit
from jax.nn import relu as jrelu
from jax.tree_util import Partial
from jax.config import config

config.update("jax_enable_x64", True) 

# %% ../02_delaunay_simulation.ipynb 6
from typing import Union, Dict, List, Tuple, Iterable, Callable, Literal
from nptyping import NDArray, Int, Float, Bool, Shape

from fastcore.foundation import patch

# %% ../02_delaunay_simulation.ipynb 10
def rot_mat(theta: float) -> NDArray[Shape["2,2"],Float]:
    """Get rotation matrix from angle in radians."""
    return np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])

def shear_mat(s: float) -> NDArray[Shape["2,2"],Float]:
    """Get shear matrix diag(s, 1/s)"""
    return np.array([[s, 0],[0, 1/s]])

def scale_mat(s: float) -> NDArray[Shape["2,2"],Float]:
    """Get shear matrix s*Id"""
    return np.diag([s,s])

# %% ../02_delaunay_simulation.ipynb 11
def get_triangular_lattice_convex(nx: int, ny: int) -> NDArray[Shape["2,*"],Float]:
    """Get (points from a) convex patch of triangular lattice of size nx*ny. nx,ny odd."""
    assert ny%2 and nx%2

    max_ny = 2*(nx-1)+1
    ny = min(ny, max_ny)
    
    y = np.arange(0, ny)*np.sqrt(3)/2
    x = np.arange(nx).astype(float)
    X, Y = np.meshgrid(x, y)

    X -= X.mean()+1/2; Y -=Y.mean()

    X = (X.T+(np.arange(ny)%2)/2).T
    pts = np.stack([X, Y]).reshape((2,nx*ny))
    
    theta = np.pi/3
    epsilon = 1e-4
    thr = floor(ny/4)
    halfplanes = [np.array([np.sin(theta), np.cos(theta)]),
                  np.array([np.sin(-theta), np.cos(-theta)])]
    vals = halfplanes[0].dot(pts)
    in_0_row = vals[np.abs(pts[1]) < epsilon]    
    is_convex = (vals > in_0_row.min()-epsilon) & (vals < in_0_row.max()+epsilon)
    
    vals = halfplanes[1].dot(pts)
    in_0_row = vals[np.abs(pts[1]) < epsilon]    
    is_convex &= (vals > in_0_row.min()-epsilon) & (vals < in_0_row.max()+epsilon)
    
    pts = pts[:, is_convex]
    pts = (pts.T-pts.mean(axis=1)).T
    return pts

def get_tri_hemesh(nx=7, ny=11, noise: float=0) -> msh.HalfEdgeMesh:
    """Create a half edge mesh of a convex patch of triangular lattice + noise."""
    pts = get_triangular_lattice_convex(nx, ny)
    tri = spatial.Delaunay(pts.T)
    mesh = msh.HalfEdgeMesh(msh.ListOfVerticesAndFaces(tri.points, tri.simplices))
    mesh.transform_vertices(lambda x: x+np.random.normal(scale=noise, size=(2,)))
    return mesh

# %% ../02_delaunay_simulation.ipynb 13
def get_triangular_lattice(nx: int, ny: int) -> Tuple[NDArray[Shape["2,*"],Float], NDArray[Shape["*"],Bool]]:
    """
    Get points for rectangular patch of triangular lattice with nx, ny points.
    
    Also return a mask which delinates bdry vertices."""

    y = np.arange(0, ny)*np.sqrt(3)/2
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

def create_rect_mesh(nx: int, ny: int, noise: float=0, defects=(0,0),
                     straight_bdry=False) -> msh.HalfEdgeMesh:
    """
    Create a half-edge mesh rectangular patch of triangular lattice. 
    
    Edges have length 1 by default. Optionally, add noise to vertex positions and create point defects
    at random positions.
    
    Parameters
    ----------
    
    nx, ny: int
        x- and y- dimensions of lattice patch
    noise: float
        Standard deviation of Gaussian noise to be added to vertex positions
    defects: tuple (int, int)
        Number of missing/duplicate defects
    straight_bdry: bool
        Keep triangles at boundary, which are not equilateral.
        
    Returns
    -------
    mesh: msh.HalfEdgeMesh
    """
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
    mesh = msh.HalfEdgeMesh(msh.ListOfVerticesAndFaces(tri.points, simplices))
    
    return mesh

# %% ../02_delaunay_simulation.ipynb 15
def create_rect_mesh_angle(nx: int, ny: int, angle=0, noise=0, max_l=1.2) -> msh.HalfEdgeMesh:
    """
    Create a half-edge mesh rectangular patch of tri lattice, with given angle between y- and lattice axis.

    This differs from create_rect_mesh by allowing to specify the angle 0°-60° between the lattice axis
    and the y-axis. Note that the resulting boundaries are typically not clean.
    
    Edges have length 1 by default. Optionally, add noise to vertex positions.
    
    Parameters
    ----------
    
    nx, ny: int
        x- and y- dimensions of lattice patch
    noise: float
        Standard deviation of Gaussian noise to be added to vertex positions
    max_l: float
        Remove faces with an edge of length > max_l. Non-equilateral triangles can occur at mesh boundary
        
    Returns
    -------
    mesh: msh.HalfEdgeMesh
    """
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
    mesh = msh.HalfEdgeMesh(msh.ListOfVerticesAndFaces(tri.points, simplices))

    mesh.transform_vertices(lambda v: v+np.random.normal(scale=noise))
    
    return mesh


# %% ../02_delaunay_simulation.ipynb 18
def get_inertia(pts: NDArray[Shape["*,2"],Float], q=0) -> NDArray[Shape["2,2"],Float]:
    """Get inertia tensor of point cloud. q in [0, 1) removes points with outlier x/y coordinates"""
    pts -= trim_mean(pts,q, axis=0)
    x, y = pts.T
    Ixx = trim_mean(x**2, q)
    Ixy = trim_mean(x*y, q)
    Iyy = trim_mean(y**2, q)
    return np.array([[Ixx, Ixy], [Ixy,Iyy]])

# %% ../02_delaunay_simulation.ipynb 20
@patch
def get_vertex_angles(self: msh.HalfEdgeMesh, method: Literal["real", "dual"]="real",
                      exclude: [None, List[int]]=None) -> Dict[int, NDArray[Shape["3"],Float]]:
    """Get dictionary of vertex angles."""
    exclude = [exclude] if exclude is None else exclude
    if method == "dual":
        lengths = {fc._fid: np.array([np.linalg.norm(x.vertices[1].coords-x.vertices[0].coords)
                                      for x in fc.hes])
                   for fc in self.faces.values() if (not (fc._fid in exclude))}
        angles = {key: tns.sides_angles(val) for key, val in lengths.items()}
    if method == "real":
        angles = []
        for fc in self.faces.values():
            if (not (fc._fid in exclude)) and (not fc.is_bdr()):
                vecs = np.stack([he.twin.face.dual_coords-fc.dual_coords for he in fc.hes])
                angle = [np.pi-tns.vectors_angle(x, y) for x,y in zip(vecs, np.roll(vecs, 1, axis=0))]
                angles[fc._fid] = np.array(angle)
    
    return angles

# %% ../02_delaunay_simulation.ipynb 25
def get_conformal_transform(mesh1: msh.HalfEdgeMesh, mesh2: msh.HalfEdgeMesh) -> callable:
    """
    Get rotation+scaling+translation to match mesh2's triangulation to mesh1. Preserves overall area.
    """
    bdry1 = np.stack([he.vertices[0].coords for he in mesh1.get_bdry_hes()])
    bdry2 = np.stack([he.vertices[0].coords for he in mesh2.get_bdry_hes()])
    rescale = np.sqrt(msh.polygon_area(bdry1)/msh.polygon_area(bdry2))

    pts1 = np.stack([v.coords for v in mesh1.vertices.values()])
    pts2 = np.stack([v.coords for v in mesh2.vertices.values()])
    mean1, mean2 = (pts1.mean(axis=0), pts2.mean(axis=0))

    rotation = orthogonal_procrustes(rescale*(pts2-mean2), pts1-mean1)[0]

    return lambda x: rotation.T@(rescale*(x-mean2))+mean1


# %% ../02_delaunay_simulation.ipynb 39
import ipywidgets as widgets
from matplotlib import animation, rc
