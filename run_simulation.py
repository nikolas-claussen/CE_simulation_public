## import stuff

import CE_simulation.mesh as msh
import CE_simulation.tension as tns
import CE_simulation.delaunay as dln
import CE_simulation.isogonal as iso
import CE_simulation.drosophila as drs

import os
import time
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm import tqdm

from copy import deepcopy
import pickle

import jax.numpy as jnp
from jax import grad as jgrad
from jax import jit
from jax.tree_util import Partial

from jax.config import config
config.update("jax_enable_x64", True) # 32 bit leads the optimizer to complain about precision loss
config.update("jax_debug_nans", False) # useful for debugging, but makes code slower!

## set save directory

save_dir = 'runs/script_large_high_tri_reg/'
try:
    os.mkdir(save_dir)
except FileExistsError:
    print('warning: save directory exists')
    
## copy script to the save directory as a way to keep track of paremters

copied_script_name = time.strftime("%Y-%m-%d_%H%M") + '_' + os.path.basename(__file__)
shutil.copy(__file__, save_dir + os.sep + copied_script_name)

## create mesh

mesh_initial, bdry_list, property_dict = drs.create_rect_initial(28, 40, noise=0.1, initial_strain=0.08,
                                                                 orientation='orthogonal', isogonal=.2,
                                                                 boundaries=None, # ['top', 'bottom']
                                                                 w_passive=3, random_seed=3)
mesh_initial.save_mesh(f"{save_dir}/initial_mesh", save_attribs=True)

## plot intial condition for reference

edge_colors = {key: "tab:grey" for key in property_dict['passive_edges']}
cell_alpha = .5
cell_colors = {key: np.hstack([drs.fridtjof_colors[val % drs.fridtjof_colors.shape[0]], [cell_alpha]])
               for key, val in property_dict['initial_row_dict'].items()}

fig = plt.figure(figsize=(8,8))
mesh_initial.cellplot(edge_colors=edge_colors, cell_colors=cell_colors)
plt.gca().set_aspect("equal", adjustable="box");
plt.xlim([-property_dict['bdry_x']-.5, property_dict['bdry_x']+.5])
plt.ylim([-property_dict['bdry_y']-.5, property_dict['bdry_y']+.5])
plt.savefig(f"{save_dir}/initial_cond.pdf")

## set parameters

### feedback parameters

m = 4
k = .5
k_cutoff = .3 # regularization term, 0.25
 
passive_ids = property_dict['passive_faces']
def params_pattern(fid):
    if fid in passive_ids:
        return {"k": .25, "m": 1}
    return {"k": k, "m": 4, "k_cutoff": k_cutoff}

params_no_pattern = {"k": k, "m": m, "k_cutoff": k_cutoff}
    
dt = .001 # time step
n_steps = 1500
forbid_reflip = 50
minimal_l = .075 # minimal edge length, lower edge lengths trigger T1

tri_mod_area = .025 # triangle area regularization, 0.01

### cell shape parameters

tol, maxiter = (1e-4, 50000)
mod_bulk = 1
mod_shear = .5
angle_penalty = 1000
bdry_penalty = 0 #5000

epsilon_l = 1e-3

A0 = jnp.sqrt(3)/2
mod_area = 0

bdr_weight = 2

passive_cells = property_dict['passive_cells']
rel_elastic_modulus = .8 # reduction in elastic modulus in passive cells
cell_id_to_modulus = np.vectorize(lambda x: 1-rel_elastic_modulus*(x in passive_cells))

use_voronoi = False # don't do shape optimization, run voronoi instead

### rest shape relaxation

k_rest = 2
def rhs_rest_shape(v):
    """Rest shape relaxation but do not relax area, i.e. trace. Also, only relax passive cells"""
    #if v._vid in property_dict['passive_cells']:
    #    delta = v.rest_shape-v.get_vrtx_shape_tensor()
    #    return -k_rest*(delta - np.trace(delta)/2 * np.eye(2))
    #else:
    #    return 0
    return -k_rest * (v.rest_shape - np.sqrt(3)*np.eye(2))
    
### package all into a single dict to pass to the optimizer method

energy_args = {"mod_bulk": mod_bulk, "mod_shear": mod_shear,
               "angle_penalty": angle_penalty, "bdry_penalty": bdry_penalty, "epsilon_l": epsilon_l,
               "A0": A0, "mod_area": mod_area}
optimizer_args = {'bdry_list': bdry_list, 'energy_args': energy_args, 'cell_id_to_modulus': cell_id_to_modulus,
                  'tol': tol, 'maxiter': maxiter, 'verbose': True, 'bdr_weight': bdr_weight}


## simulation loop

# note that we don't keep a list of all meshes, because that can consume a lot of memory for long simulations

print_T1s = True

times = [0]
last_flipped_edges = [[]]
mesh = deepcopy(mesh_initial)
mesh_previous = deepcopy(mesh_initial)

for i in tqdm(range(0, n_steps)):
    # euler step
    mesh.euler_step(dt=dt, rhs_tension=tns.excitable_dt_act_pass, params=params_pattern,
                    rhs_rest_shape=rhs_rest_shape)
    # flatten triangulation
    mesh.flatten_triangulation(mod_area=tri_mod_area, tol=1e-4)
    # primal optimization
    if use_voronoi:
        mesh.set_voronoi()
    else:
        mesh.optimize_cell_shape(**optimizer_args)
    # check for intercalation
    flipped, failed_flip = mesh.intercalate(exclude=list(msh.flatten(last_flipped_edges[-forbid_reflip:])),
                                            minimal_l=minimal_l, reoptimize=True, optimizer_args=optimizer_args)
    if print_T1s:
	    if failed_flip
            print(f"tpt {i}: flip {flipped}, failed {failed_flip}")
	    else:
            print(f"tpt {i}: flip {flipped}")
    # rescale & reorient triangulation
    mesh.transform_vertices(dln.get_conformal_transform(mesh_previous, mesh))
    # log & save
    last_flipped_edges.append(flipped)
    times.append(times[-1]+dt)
    mesh.save_mesh(f"{save_dir}/{str(i).zfill(4)}_mesh", save_attribs=True)
    mesh_previous = deepcopy(mesh)
last_flipped_edges.append([])

pickle.dump(last_flipped_edges, open(f"{save_dir}T1s.p", "wb"))
np.savetxt(f"{save_dir}times.txt", np.array(times))
