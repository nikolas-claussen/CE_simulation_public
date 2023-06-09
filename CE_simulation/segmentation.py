# AUTOGENERATED! DO NOT EDIT! File to edit: ../07_from_segmentation.ipynb.

# %% auto 0
__all__ = ['prepare_input', 'for_imshow', 'get_com_dict', 'image_to_hmesh']

# %% ../07_from_segmentation.ipynb 2
import CE_simulation.mesh as msh
import CE_simulation.tension as tns
import CE_simulation.delaunay as dln
import CE_simulation.isogonal as iso
import CE_simulation.drosophila as drs

# %% ../07_from_segmentation.ipynb 3
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import optimize, ndimage, sparse

from tqdm.notebook import tqdm

from copy import deepcopy
import pickle

# %% ../07_from_segmentation.ipynb 4
from typing import Union, Dict, List, Tuple, Iterable, Callable, Any
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../07_from_segmentation.ipynb 5
import jax.numpy as jnp
from jax import jit
import jax
from jax.tree_util import Partial

from jax.config import config
config.update("jax_enable_x64", True) # 32 bit leads the optimizer to complain about precision loss
config.update("jax_debug_nans", False) # useful for debugging, but makes code slower!

# %% ../07_from_segmentation.ipynb 6
import ipywidgets as widgets
import functools
from matplotlib import animation, rc

# %% ../07_from_segmentation.ipynb 7
from skimage import measure, segmentation, future
import networkx as nx

# %% ../07_from_segmentation.ipynb 13
def prepare_input(img, min_area=4):
    """
    Prepare input image data for computation of cell adjacency graph.
    
    If the input image is a binary image, it is assumed to be a segmentation, and a labled image
    is created by labeling the connected components of the image. If it is not (i.e. the image contains
    more than two values), this step is skipped.
    
    Next, the image is manipulated to remove very small regions and ensure that all regions are connected
    (according to pixel connectivity 1, i.e. nearest-neighbors-no-diagonals). Non-connected components
    and small regions are deleted and filled by expanding the surrounding regions. This is done to avoid
    potential bugs when computing the neighborhood graph using morphological operations.
    
    Parameters
    ----------
    img : 2d np.array
        Either labeled image (integer type), or segmentation (binary)
    min_area : int, default 4
        Regions with smaller area are removed
    
    Returns
    -------
    lbl : 2d np.array of ints
        Sanitized labeled array
    """
    if len(np.unique(img)) == 2: # segmentation
        lbl = ndimage.label(img==0)[0]
        lbl = expand_labels(lbl, distance=2)
    else:
        lbl = img
    # remove small regions
    for key in np.unique(lbl):
        if (lbl==key).sum() == min_area:
            lbl[lbl==key] = 0 
    lbl = expand_labels(lbl, distance=2)
    # now ensure each remaining region is connected - remove all but the largest connected component.
    for key in np.unique(lbl):
        sublabled = ndimage.label(lbl==key)[0]
        subareas = {key: (sublabled==key).sum() for key in np.arange(1, np.max(sublabled)+1)}
        max_subarea = max(subareas, key=subareas.get)
        lbl[(sublabled != max_subarea) & (lbl == key)] = 0
    lbl = expand_labels(lbl, distance=2)
    return lbl

# %% ../07_from_segmentation.ipynb 17
def for_imshow(labeled: NDArray[Shape["*,*"], Int], colors: NDArray[Shape["*,3"], Float]=drs.fridtjof_colors
              ) -> NDArray[Shape["*,*,3"], Float]:
    """Create colored array from labeled image using array of RBG colors, with black boundaries."""
    bdry = segmentation.find_boundaries(labeled, mode='outer')
    rgb = colors[labeled % colors.shape[0]]
    rgb = (rgb.transpose((2,0,1)) * (bdry == 0)).transpose((1,2,0))
    return rgb

# %% ../07_from_segmentation.ipynb 21
def _add_edge_filter(values, graph_dict):
    """Create edge in `graph_dict` between central element of `values` and the rest.
    Add an edge between the middle element in `values` and
    all other elements of `values` into `graph`.  ``values[len(values) // 2]``
    is expected to be the central value of the footprint used.
    Parameters
    ----------
    values : array
        The array to process.
    graph_dict : dict
        The graph to add edges in.
    Returns
    -------
    0 : float
        Always returns 0. The return value is required so that `generic_filter`
        can put it in the output array, but it is ignored by this filter.
    """
    values = values.astype(int)
    center = values[len(values) // 2]
    for value in values:
        if value != center:
            graph_dict[center].add(value)
    return 0.

# %% ../07_from_segmentation.ipynb 22
def _4fold_filter(values, graph_dict):
    """
    Create edges in `graph_dict` for 4-fold vertices which meet in the square configuration.
    
    To avoid 
    
    Parameters
    ----------
    values : array
        The array to process. Should to be of shape 4 from a 2x2 strelm
    graph_dict : dict
        The graph to add edges in.
    Returns
    -------
    0 : float
        Always returns 0. The return value is required so that `generic_filter`
        can put it in the output array, but it is ignored by this filter.
    """
    values = values.astype(int)
    if np.unique(values).size == values.size:
        graph_dict[values[0]].add(values[3])
        graph_dict[values[3]].add(values[0])
    return 0.

# %% ../07_from_segmentation.ipynb 39
def get_com_dict(labeled: NDArray[Shape["*,*"], Int]) -> Dict[int, NDArray[Shape["2"], Float]]: 
    """Get the centroids of regions in a labeled array"""
    return {key: np.array(ndimage.center_of_mass(labeled==key)) for key in np.unique(labeled)}
        

# %% ../07_from_segmentation.ipynb 54
def image_to_hmesh(img, min_area=4, vertex_dil=1, cell_size=None):
    """
    Compute half-edge mesh from image data.
    
    Input can be either a segmentation or a labeled image. Output is a half-edfge mesh, whose vertices
    are the cells in the image data. The primal vertex positions are found from the image data. The dual
    vertex positions are set to the cell centroids.
    
    The image is pre-processed to sanitize the segmentation and avoid errors in the cell adjacency graph.
    
    Currrently, the algorithm is not very fast since it uses an inefficient method for vertex position detection.
    
    
    Parameters
    ----------
    img : 2d np.array
        Image, either binary (in which case it is assumed to be a segmentation), or int (in which case it is
        assumed to show cell labels)
    min_area : int, default 4
        Minimum area for cells. Very small cells can lead to bugs
    vertex_dil : int, default 1
        Number of dilations used to determine vertex positions
    cell_size : int
        Approximate cell diameter for croppimg image when doing vertex detection. If None, is set automatically.
    
    Returns
    -------
    hemesh : HalfEdgeMesh
        Half edge mesh representing the image data
    
    """
    # sanitize input, convert segmentation to labeld if required
    lbl = prepare_input(img)
    
    # compute adjaceny graph and create half-edge mesh
    tris = get_triangles(labeled_to_graph(lbl))
    points = get_com_dict(lbl)
    points = {key: val[::-1] for key, val in points.items()}
    
    hemesh = msh.HalfEdgeMesh(msh.ListOfVerticesAndFaces(points, tris))

    if cell_size is None:
        cell_size = int(np.ceil(np.quantile(list(hemesh.get_edge_lens().values()), .95)))
    
    # set vertex positions
    for fc in hemesh.faces.values():
        center = np.round(fc.coords.mean(axis=0)).astype(int)
        # use take to wrap around edges
        windowed = labeled_test.take(range(center[1]-cell_size, center[1]+cell_size), mode='wrap', axis=0).take(
                                     range(center[0]-cell_size,center[0]+cell_size),  mode='wrap', axis=1)
        com = ndimage.center_of_mass(np.prod([ndimage.binary_dilation(windowed==v._vid, iterations=1)
                                              for v in fc.vertices], axis=0))
        fc.primal_coords = center + com[::-1] - np.array([cell_size, cell_size])    
        
    return hemesh
