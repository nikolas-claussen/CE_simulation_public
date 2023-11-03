# AUTOGENERATED! DO NOT EDIT! File to edit: ../09c_disorder_phase_diagram_analysis.ipynb.

# %% auto 0
__all__ = ['get_width_height', 'get_delta_centerline', 'get_anisos_S', 'get_anisos_T', 'get_excess_aniso', 'make_segments3d',
           'colorline3d']

# %% ../09c_disorder_phase_diagram_analysis.ipynb 3
import jax
import jax.numpy as jnp
from jax import grad as jgrad
from jax import jit
from jax.config import config

jax.default_device(jax.devices('cpu')[0])
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)
config.update("jax_platform_name", "cpu")

# %% ../09c_disorder_phase_diagram_analysis.ipynb 4
import CE_simulation.mesh as msh
import CE_simulation.tension as tns
import CE_simulation.delaunay as dln
import CE_simulation.isogonal as iso
import CE_simulation.drosophila as drs
import CE_simulation.disorder as dis
import CE_simulation.hessian as hes
import CE_simulation.ecmc as ecm

# %% ../09c_disorder_phase_diagram_analysis.ipynb 5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from tqdm.notebook import tqdm

from scipy import spatial, ndimage
from skimage.transform import downscale_local_mean

from collections import Counter
import itertools

from copy import deepcopy
import os
import pickle

# %% ../09c_disorder_phase_diagram_analysis.ipynb 6
from joblib import Parallel, delayed
import gc

# %% ../09c_disorder_phase_diagram_analysis.ipynb 7
from typing import Sequence, Union, Dict, List, Tuple, Iterable, Callable, Any
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../09c_disorder_phase_diagram_analysis.ipynb 10
def get_width_height(mesh, q=0.8, exclude=None):
    """Get tissue extension by means of q% x- and y-axis bounding box"""
    exclude = [] if exclude is None else exclude
    centroids = np.stack([v.get_centroid() for key, v in mesh.vertices.items()
                           if (not v.is_bdry()) and (not v._vid in exclude)])

    delta = np.quantile(centroids, axis=0, q=q)-np.quantile(centroids, axis=0, q=1-q,)
    return delta


def get_delta_centerline(meshes, sigma=2, add_start_pt=True):
    """Get tissue extension using the centerline measure."""
    start, stop = ecm.get_left_right_pt(meshes[0])
    top_bdry, bottom_bdry = ecm.get_top_bottom_bdry(meshes[0], tol=4)

    
    centerlines = []
    lengths = []
    widths = []

    for m in tqdm(meshes):
        centerlines.append(ecm.get_centerline(m, start, stop, dx=0.5, sigma=2))
        lengths.append(ecm.get_arclen(centerlines[-1]))
        widths.append(ecm.get_width(m, top_bdry, bottom_bdry))

    delta = np.log(np.stack([np.array(lengths)/lengths[0], np.array(widths)/widths[0]], axis=-1))
    if add_start_pt:
        delta = np.vstack([[0,0], delta])
    if sigma > 0:
        delta_smooth = ndimage.gaussian_filter1d(delta, sigma, axis=0)
    return delta_smooth, centerlines

# %% ../09c_disorder_phase_diagram_analysis.ipynb 11
def get_anisos_S(mesh):
    """Compute single-triangle and mean anisotropy based on the quadratic S=T.T tensor"""
    tensors = np.stack([fc.get_stress_tensor() for fc in mesh.faces.values()])
    tensors = (tensors.T / np.linalg.eigvalsh(tensors).sum(axis=1)).T # normalize

    vals_all = np.linalg.eigvalsh(tensors)
    anisotropy_all = np.mean(vals_all[:,1] - vals_all[:,0])

    vals_mean = np.linalg.eigvalsh(tensors.mean(axis=0))
    anisotropy_mean = vals_mean[1] - vals_mean[0]

    return anisotropy_mean, anisotropy_all
def get_anisos_T(mesh):
    """Compute single-triangle and mean anisotropy based on the singular values of the linear T tensor"""
    T_tensors = np.stack([val for val in mesh.get_T_tensor().values()])
    U, S, Vh = np.linalg.svd(T_tensors) # we want Vh I think

    S = (2*S.T / S.sum(axis=1)).T # normalize S
    anisotropy_all = np.mean(S[:,1] - S[:,0])
    
    S_diag = np.einsum('ij,nj->nij', np.eye(2), S)
    aniso_tensors = np.einsum('nji,njk,nkl->nil', Vh, S_diag, Vh)
    anisotropy_mean = np.linalg.eigvalsh(aniso_tensors.mean(axis=0))
    anisotropy_mean = anisotropy_mean[1] - anisotropy_mean[0]
    
    return anisotropy_all, anisotropy_mean

# %% ../09c_disorder_phase_diagram_analysis.ipynb 12
def get_excess_aniso(mesh):
    """Compute 'excess' single triangle anisotropy"""
    tensors = np.stack([fc.get_stress_tensor() for fc in mesh.faces.values()])
    tensors = (tensors.T / np.linalg.eigvalsh(tensors).sum(axis=1)).T # normalize

    # subtract trace
    tensors = tensors-np.einsum('vii,jk->vjk', tensors, np.eye(2))/2
    tensors_mean = tensors.mean(axis=0)
    
    tensors_nomean = tensors - tensors_mean
    tensors_norm = (2*np.linalg.norm(tensors, axis=(1,2), ord=2)**2).mean()
    tensors_mean_norm = (2*np.linalg.norm(tensors_mean, axis=(0,1), ord=2)**2)
    tensors_nomean_norm = (2*np.linalg.norm(tensors_nomean, axis=(1,2), ord=2)**2).mean()
    
    return np.sqrt(tensors_norm), np.sqrt(tensors_mean_norm), np.sqrt(tensors_nomean_norm)

# %% ../09c_disorder_phase_diagram_analysis.ipynb 13
def make_segments3d(x, y, z):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline3d(x, y, z, cmap='cool', norm=plt.Normalize(0.0, 1.0),
              linewidth=3, alpha=1.0, ax=None):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    segments = make_segments3d(x, y, z)
    arr = np.linspace(0.0, 1.0, len(x))

    # Default colors equally spaced on [0,1]:
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, array=arr,
                          linewidth=linewidth, alpha=alpha)
    if ax is None:
        ax = plt.gca()
    ax.add_collection3d(lc)
        
    return lc