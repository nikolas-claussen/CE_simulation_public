###############################################
# import the relevant libraries
###############################################

import CE_simulation.mesh as msh
import CE_simulation.tension as tns
import CE_simulation.delaunay as dln
import CE_simulation.isogonal as iso
import CE_simulation.drosophila as drs
import CE_simulation.disorder as dis
import CE_simulation.hessian as hes
import CE_simulation.ecmc as ecm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from collections import Counter
import itertools

from copy import deepcopy
import os
import sys
import pickle
import time as time_module

from joblib import Parallel, delayed

import jax.numpy as jnp
from jax import grad as jgrad
from jax import jit

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)


###############################################
# define local functions if necessary
###############################################



###############################################
# set parameters
###############################################

# the parameters defined here will be the same for all simulations launched from this script

# define timeout for optimizer step, in s. If an optimizer step takes longer than this, kill the function after.
timeout_iteration = 20*60 

# initial condition parameters
n_x = 23
n_y = 34

initial_strain = 0.2
initial_strain = np.sqrt(1+initial_strain)-1 # due to convention
noise_gaussian = 0.1

spread_passive = 0.1

# triangulation dynamics parameters
m = 4
k = 6
k_cutoff = 0.3
k_relax = 2

# elastic energy parameters
mod_bulk = 1
mod_shear = 1
angle_penalty = 1000
bdry_penalty = 5000

rel_bulk_modulus = -20 # reduction in bulk modulus in passive cells
rel_shear_modulus = -20 # reduction in shear modulus in passive cells
rel_angle_penalty = 0.995

# numerical parameters
dt = .005 # time step
forbid_reflip = 20
tri_mod_area = .01 # triangle area regularization
minimal_l = 0.06
tol, maxiter = (1e-4, 100000)
epsilon_l = (1e-3, 1e-3)  # mollifying parameters to ensure energy differentiability
bdr_weight = 1 # avoid problems at boundary

def rhs_rest_shape(v): # no rest shape relaxation
    return 0

# package all into a single dict to pass to the optimizer method
energy_args = {'mod_bulk': mod_bulk, 'mod_shear': mod_shear,
               'angle_penalty': angle_penalty, 'bdry_penalty': bdry_penalty, 'epsilon_l': epsilon_l,
               'mod_area': 0, 'mod_perimeter': 0}
optimizer_args = {'energy_args': energy_args, 'tol': tol, 'maxiter': maxiter, 'verbose': True, 'bdr_weight': bdr_weight}

# directory for saving stuff
base_dir = "/data/Nikolas/GBE_simulation/runs/germ_band/"

###############################################
# define local function for running simulation
###############################################

def run_phase_diagm_sim(w_passive, n_steps, random_seed=None, run_id=0):
    """
    Local function to run simulation in parallel.
    
    run_id can be used to additionally label the folder for saving, in case multiple simulations are run
    for the same parameter values.
    """
    
    # define directiory for saving stuff, save "metadata"
    save_dir = f"{base_dir}/{run_id}/w_passive_{w_passive}_randomSeed_{random_seed}"
    try:
        os.makedirs(save_dir, exist_ok=True)
    except FileExistsError:
        print('Warning: directory exists')
    drs.save_self(save_dir, fname=None)
    logfile = save_dir + "/" + time_module.strftime("%Y-%m-%d_%H%M") + "log.txt"
    open(logfile, 'a').close()
    
    # set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # create initial condition
    mesh_initial, property_dict = drs.create_rect_initial(24, 34, noise=noise_gaussian, initial_strain=initial_strain,
                                                          orientation='orthogonal', isogonal=0, random_seed=random_seed,
                                                          boundaries=['top', 'bottom',], w_passive=w_passive, w_passive_lr=0.5)
    for e in property_dict['passive_edges']:     # set passive tensions in passive region:
        #mesh_initial.hes[e].passive = mesh_initial.hes[e].rest
        #mesh_initial.hes[e].twin.passive = mesh_initial.hes[e].twin.rest
        mesh_initial.hes[e].passive = mesh_initial.hes[e].twin.passive = np.random.normal(loc=1, scale=spread_passive)
    # save initial condition as a plot
    edge_colors = {key: "tab:grey" for key in property_dict['passive_edges']}
    cell_colors = {key: mpl.colors.to_rgba('tab:blue') * np.array([1,1,1,.5])
                   for key in mesh_initial.vertices.keys() if not key in property_dict['passive_cells']}
    fig = plt.figure(figsize=(8,8))
    mesh_initial.cellplot(edge_colors=edge_colors, cell_colors=cell_colors)
    plt.xlim([-2*property_dict['bdry_x']-.5, 2*property_dict['bdry_x']+.5])
    plt.ylim([-property_dict['bdry_y']-.5, property_dict['bdry_y']+.5])
    plt.gca().set_aspect("equal", adjustable="box");
    plt.axis("off")
    plt.savefig(f"{save_dir}/plot_initial.pdf")
    plt.close()
    with open(logfile, 'a') as f:
        f.write(f"{-1}\tGenerated initial condition\n")
    with open(f'{save_dir}/property_dict.pickle', 'wb') as handle:
        pickle.dump(property_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # patterning of mechanics
    cell_id_to_modulus = defaultdict(lambda: (lambda x: 1))
    cell_id_to_modulus["mod_bulk"] = lambda x: 1-rel_bulk_modulus*(x in property_dict['passive_cells'])
    cell_id_to_modulus["mod_shear"] = lambda x: 1-rel_shear_modulus*(x in property_dict['passive_cells'])
    # also reduce the angle penalty proportionally in the passive region
    edge_id_to_angle_penalty = lambda x: 1-rel_angle_penalty*(x in property_dict['passive_edges'])
    optimizer_args['cell_id_to_modulus'] = cell_id_to_modulus
    optimizer_args['edge_id_to_angle_penalty'] = edge_id_to_angle_penalty
    
    def params_pattern(fid):
        if fid in property_dict['passive_faces']:
            return {"m": None, "k_cutoff": None, "k": k, "is_active": False, "subtract_passive": False} 
        return {"m": m, "k_cutoff": k_cutoff, "k": k, "is_active": True, "subtract_passive": False}
        
    # main simulation loop
    times = [0]; last_flipped_edges = [[]]
    mesh = mesh_initial
    mesh_previous = deepcopy(mesh)
    # save 0th timepoint
    mesh.save_mesh(f"{save_dir}/{str(0).zfill(4)}_mesh", save_attribs=False) # can set that to false.

    for i in range(n_steps-1):
        # euler step
        current_time = time_module.time() # excitable_dt_act_pass_perimeter
        mesh.euler_step(dt=dt, rhs_tension=tns.excitable_dt_act_pass_new_passive_rest, params=params_pattern,
                        rhs_rest_shape=rhs_rest_shape)
        # flatten triangulation.
        sol = mesh.flatten_triangulation(mod_area=tri_mod_area, tol=1e-4, return_sol=True, reset_intrinsic=False,
                                         soften_direct=0, soften_indirect=0)
        # cancel execution if this fails. precision loss and insufficient iteration errors are typically harmless.
        with open(logfile, 'a') as f:
            f.write(f"{i}\tTriangulation flattening: {sol.message}\n")
        if sol["status"] !=0 and (sol["message"] == "NaN result encountered."):
            break
        # relax intrinsic lengths
        if k_relax < np.infty:
            mesh.euler_step_relax(k_relax=k_relax/dt, dt=dt)
        else:
            mesh.set_rest_lengths()
        # primal optimization
        sol = mesh.optimize_cell_shape(**optimizer_args, return_sol=True)
        with open(logfile, 'a') as f:
            f.write(f"{i}\tCell shape optimization: {sol.message}\n")
        if sol["status"] !=0 and (sol["message"] == "NaN result encountered."):
            break
        # check for intercalation
        flipped, failed_flip = mesh.intercalate(exclude=list(msh.flatten(last_flipped_edges[-forbid_reflip:])),
                                                minimal_l=minimal_l, reoptimize=False,
                                                optimizer_args=optimizer_args)
        # set passive tension on passive edges
        for e in flipped:
            if e in property_dict['passive_edges']:
                mesh.hes[e].passive = mesh.hes[e].twin.passive = np.random.normal(loc=1, scale=spread_passive)

        # rescale & reorient triangulation
        mesh.transform_vertices(dln.get_conformal_transform(mesh_previous, mesh))
        # log & save
        last_flipped_edges.append(flipped)
        times.append(times[-1]+dt)
        mesh_previous = deepcopy(mesh)
        mesh_previous.save_mesh(f"{save_dir}/{str(i+1).zfill(4)}_mesh", save_attribs=False)
        # time out if necessary
        if (time_module.time()-current_time) > timeout_iteration:
            with open(logfile, 'a') as f:
                f.write(f"{i}\tTimed out (took {round(time_module.time()-current_time)} s), exiting\n")
            break
    
    # save list of T1s
    last_flipped_edges.append([])
    with open(f'{save_dir}/last_flipped_edges.pickle', 'wb') as handle:
        pickle.dump(last_flipped_edges, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save final mesh as a plot
    fig = plt.figure(figsize=(8,8))
    mesh_previous.cellplot(edge_colors=edge_colors, cell_colors=cell_colors)
    plt.xlim([-2*property_dict['bdry_x']-.5, 2*property_dict['bdry_x']+.5])
    plt.ylim([-property_dict['bdry_y']-.5, property_dict['bdry_y']+.5])
    plt.gca().set_aspect("equal", adjustable="box");
    plt.axis("off")
    plt.savefig(f"{save_dir}/plot_final.pdf")
    plt.close()

    return None

###############################################
# define simulation sweep
###############################################

n_steps = 250 # 125
w_passives = [0.5, 1.5]
reps = np.array([1, 2, 3]) # number of replicates
to_do = list(itertools.product(w_passives, reps))

run_id = "germ_band_new_post_T1_TollRM9_new_passive_long"

###############################################
# run simulation
###############################################

Parallel(n_jobs=6, prefer=None)(delayed(run_phase_diagm_sim)( 
    w_passive, n_steps=n_steps, random_seed=rep, run_id=run_id)
    for w_passive, rep in to_do)





