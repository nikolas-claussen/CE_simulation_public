## import stuff

import os

import numpy as np

from scipy.integrate import solve_ivp
from scipy import optimize
from scipy import linalg

from copy import deepcopy
import pickle

from fastcore.foundation import patch

import jax.numpy as jnp
from jax import grad as jgrad
from jax import jit
from jax.tree_util import Partial
from jax.config import config
from jax.nn import relu as jrelu

config.update("jax_enable_x64", True)

from CE_simulation.triangle import *
from CE_simulation.tension import *
from CE_simulation.delaunay import *
from CE_simulation.isogonal import *

from CE_simulation.boundary_jax import get_triangular_lattice, create_rect_mesh, get_centroid,\
                                       get_bdry, get_conformal_transform, get_areas,\
                                       get_primal_energy_fct_jax, get_E, get_E_jac, polygon_area,\
                                       excitable_dt_act_pass, euler_step, flatten_triangulation_jax,\
                                       get_tri_areas, get_flip_edge, optimize_cell_shape,\
                                       create_rect_mesh_angle

## create mesh

np.random.seed(2)
mesh_initial = create_rect_mesh(48, 36, noise=0.15, defects=(0, 0), straight_bdry=False)

mesh_initial.transform_vertices(rot_mat(np.pi/2))
center = np.mean([v.coords for v in mesh_initial.vertices.values()], axis=0)
mesh_initial.transform_vertices(lambda x: x-center)
mesh_initial.set_voronoi()

mesh_initial.transform_vertices(shear_mat(1.15))
mesh_initial.set_rest_lengths()

passive_ids = []
active_ids = []

max_y = np.max([val.dual_coords[1] for val in mesh_initial.faces.values()])
w_passive = 8 # 6
for fc in mesh_initial.faces.values():
    if fc.is_bdr():  # make a passive edge - let's see if that is necessary
        passive_ids.append(fc._fid)
    if np.abs(fc.dual_coords[1]) > (max_y-w_passive):
        passive_ids.append(fc._fid)
    else:
        active_ids.append(fc._fid)

passive_ids = sorted(passive_ids)
active_ids = sorted(active_ids)

passive_cells = [v._vid for v in mesh_initial.vertices.values()
                 if not v.is_bdry() and any([fc._fid in passive_ids for fc in v.get_face_neighbors()])]

max_y_cells = np.max([v.get_centroid()[1] for v in mesh_initial.vertices.values() if not v.is_bdry()])

w_bdry = .4

bdry_up_ids= []
bdry_down_ids = []

for v in mesh_initial.vertices.values():
    if (v.get_centroid()[1] > (max_y_cells-w_bdry)) and (not v.is_bdry()):
        bdry_up_ids.append(v._vid)
    if (v.get_centroid()[1] < -(max_y_cells-w_bdry)) and (not v.is_bdry()):
        bdry_down_ids.append(v._vid)
        
bdry_y = 23 # 11

def up_penalty(x):
    return (x[1]-(bdry_y))**2
def down_penalty(x):
    return (x[1]+(bdry_y))**2

up_penalty = Partial(jit(up_penalty))
down_penalty = Partial(jit(down_penalty))
bdry_list = None #([up_penalty, bdry_up_ids], [down_penalty, bdry_down_ids])

## parameters

m = 4
k = .5
k3 = .2
 
def params_pattern(fid):
    if fid in passive_ids:
        return {"k": .25, "m": 1}
    return {"k": k, "m": 4, "k3": k3}

params_no_pattern = {"k": k, "m": m, "k3": k3}
    
dt = .001 # 0.01 too large for m=4! .001
n_steps = 1300
forbid_reflip = 20
minimal_l = .075  # negative = should create overshoot.maybe too much?

tri_mod_area = .01

tol, maxiter = (1e-4, 20000) # .5*1e-4, 
mod_bulk = 1
mod_shear = .3 # .5
angle_penalty = 1000
bdry_penalty = 0 #5000

epsilon_l = 1e-3

A0 = jnp.sqrt(3)/2
mod_area = 0

bdr_weight = 2

rel_elastic_modulus = .8 # reduction in elastic modulus in passive cells
cell_id_to_modulus = np.vectorize(lambda x: 1-rel_elastic_modulus*(x in passive_cells))

energy_args = {"mod_bulk": mod_bulk, "mod_shear": mod_shear,
               "angle_penalty": angle_penalty, "bdry_penalty": bdry_penalty, "epsilon_l": epsilon_l,
               "A0": A0, "mod_area": mod_area}

use_voronoi = False # don't do shape optimization, run voronoi instead

k_rest = 4 # 2

iso = 0.25
for v in mesh_initial.vertices.values():
    v.rest_shape = np.sqrt(3) * np.array([[1-iso, 0],[0, 1+iso]])

def rhs_rest_shape(v):
    """Rest shape relaxation but do not relax area, i.e. trace. Also, only relax passive cells"""
    #if v._vid in passive_cells:
    #    delta = v.rest_shape-v.get_shape_tensor()
    #    return -k_rest*(delta - np.trace(delta)/2 * np.eye(2))
    #else:
    #    return 0
    delta = v.rest_shape-np.eye(2)  # v.get_shape_tensor()
    return -k_rest*(delta - np.trace(delta)/2 * np.eye(2))

meshes = [deepcopy(mesh_initial)]
times = [0]
last_flipped_edges = [[]]

## simulation loop

save = True
dir_name = "germ_band_very_large_isogonal_no_bdry_script_long_2"
if save:
    try:
        os.mkdir(f"runs/{dir_name}/")
    except FileExistsError:
        print('warning, directory exists')
## simulation loop

mesh = deepcopy(meshes[-1])
for i in range(n_steps):
    print(i)
    # euler step
    mesh.euler_step(dt=dt, rhs=excitable_dt_act_pass, params=params_pattern, rhs_rest_shape=rhs_rest_shape)
    # flatten triangulation
    mesh.flatten_triangulation_jax(mod_area=tri_mod_area, tol=1e-4)
    # primal optimization
    if use_voronoi:
        mesh.set_voronoi()
    else:
        mesh.optimize_cell_shape(bdry_list=bdry_list, energy_args=energy_args,
                                 cell_id_to_modulus=cell_id_to_modulus,
                                 tol=tol, maxiter=maxiter, verbose=True, bdr_weight=bdr_weight)
    # check for intercalation
    flipped = []
    failed_flip = []
    flip_edge = get_flip_edge(mesh, minimal_l=minimal_l,
                              exclude=list(flatten(last_flipped_edges[-forbid_reflip:])))
    while flip_edge is not None:
        try:
            print(f"flip {flip_edge}, tpt {i}")
            he = mesh.hes[flip_edge]
            mesh.flip_edge(flip_edge)
            mesh.reset_rest_passive_flip(he, method="smooth")
            f0, f1 = (he.face, he.twin.face)
            f0.dual_coords, f1.dual_coords = rotate_about_center(np.stack([f0.dual_coords, f1.dual_coords]))
            flipped.append(he._heid)
            if use_voronoi:
                mesh.set_voronoi()
            else:
                mesh.optimize_cell_shape(bdry_list=bdry_list, energy_args=energy_args,
                                         cell_id_to_modulus=cell_id_to_modulus,
                                         tol=tol, maxiter=maxiter, verbose=True)
            exclude = list(flatten(last_flipped_edges[-forbid_reflip:]))+flipped+failed_flip
            flip_edge = get_flip_edge(mesh, minimal_l=minimal_l, exclude=exclude)
            
        except ValueError:
            print(f"failed flip {flip_edge}, tpt {i}")
            failed_flip.append(flip_edge)
            exclude = list(flatten(last_flipped_edges[-forbid_reflip:]))+flipped+failed_flip
            flip_edge = get_flip_edge(mesh, minimal_l=minimal_l, exclude=exclude)
            continue

    # rescale & reorient triangulation
    mesh.transform_vertices(get_conformal_transform(meshes[-1], mesh))
    # log
    last_flipped_edges.append(flipped)
    meshes.append(deepcopy(mesh))
    times.append(times[-1]+dt)
    if save:
        mesh.save_mesh(f"runs/{dir_name}/{str(i).zfill(4)}_mesh.txt")
        rest_dict = {key: val.rest_shape for key, val in mesh.vertices.items()}
        pickle.dump(rest_dict, open(f"runs/{dir_name}/{str(i).zfill(4)}_rest_shape.p", "wb"))
        passive_dict = {key: val.passive for key, val in mesh.hes.items()}
        pickle.dump(passive_dict, open(f"runs/{dir_name}/{str(i).zfill(4)}_passive.p", "wb"))
        # all other info should be contained in mesh

last_flipped_edges.append([])


