# AUTOGENERATED! DO NOT EDIT! File to edit: ../04e_boundary_conditions_fake_PBC.ipynb.

# %% auto 0
__all__ = []

# %% ../04e_boundary_conditions_fake_PBC.ipynb 3
from .triangle import *
from .tension import *
from .delaunay import *
from .isogonal import *

# %% ../04e_boundary_conditions_fake_PBC.ipynb 4
import os
import sys
import importlib

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
from scipy import linalg

from tqdm.notebook import tqdm

from copy import deepcopy

from collections import Counter, defaultdict

# %% ../04e_boundary_conditions_fake_PBC.ipynb 5
from dataclasses import dataclass
from typing import Union, Dict, List, Tuple, Iterable, Callable
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../04e_boundary_conditions_fake_PBC.ipynb 6
import ipywidgets as widgets
from matplotlib import animation, rc

# %% ../04e_boundary_conditions_fake_PBC.ipynb 7
import jax.numpy as jnp
from jax import grad as jgrad
from jax import jit
from jax.tree_util import Partial
from jax.config import config
config.update("jax_enable_x64", True) # 32 bit leads the optimizer to complain about precision loss
#config.update("jax_debug_nans", True)  # useful for debugging, but makes code slower!

# %% ../04e_boundary_conditions_fake_PBC.ipynb 12
@patch
def optimize_cell_shape(self: HalfEdgeMesh, bdry_list=None,
                        energy_args=None, cell_id_to_modulus=None,
                        tol=1e-3, maxiter=500, verbose=True, bdr_weight=2):
    """Primal optimization. cell_id_to_modulus: function from _vid to relative elastic modulus"""
    x0 = mesh.dual_vertices_to_initial_cond()
    get_E_arrays, cell_list_vids = mesh.get_primal_energy_fct_jax(bdry_list)

    if energy_args is None:
        energy_args = {"mod_bulk": 1, "mod_shear": .2,"angle_penalty": 1000, "bdry_penalty": 100,
                       "epsilon_l": 1e-4}
    if cell_id_to_modulus is not None:
        mod_bulk = energy_args["mod_bulk"]*np.vectorize(cell_id_to_modulus)(cell_list_vids)
        mod_shear = energy_args["mod_shear"]*np.vectorize(cell_id_to_modulus)(cell_list_vids)
    else:
        mod_bulk, mod_shear = (energy_args["mod_bulk"], energy_args["mod_shear"])
    
    if bdr_weight != 1:
        is_bdr = np.array([any([fc.is_bdr() for fc in self.vertices[v].get_face_neighbors()])
                          for v in cell_list_vids])
        mod_bulk *= (bdr_weight*is_bdr+(1-is_bdr))
        mod_shear *= (bdr_weight*is_bdr+(1-is_bdr))
    
    cell_shape_args = (mod_bulk, mod_shear, energy_args["angle_penalty"], energy_args["bdry_penalty"],
                       energy_args["epsilon_l"])
    
    sol = optimize.minimize(get_E, x0, jac=get_E_jac, args=get_E_arrays+cell_shape_args,
                             method="CG", tol=tol, options={"maxiter": maxiter})
    if sol["status"] !=0 and verbose:
        print("Cell shape optimization failed", sol["message"])
    new_coord_dict = self.initial_cond_to_dual_vertices(sol["x"])
    for key, val in self.faces.items():
        val.dual_coords = new_coord_dict[key]
