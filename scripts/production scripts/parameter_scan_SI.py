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
n_x = 35
n_y = 30

initial_strain = 0.1
initial_strain = np.sqrt(1+2*initial_strain)-1 # due to convention

# triangulation dynamics parameters
m = 4

# elastic energy parameters
mod_bulk = 1
mod_shear = 1
angle_penalty = 1000
bdry_penalty = 0

# numerical parameters
dt = .005 # time step
forbid_reflip = 20
tri_mod_area = .01 # triangle area regularization
minimal_l = 0.06
tol, maxiter = (1e-4, 100000)
epsilon_l = (1e-3, 1e-3)  # mollifying parameters to ensure energy differentiability
bdr_weight = 2 # avoid problems at boundary
rel_elastic_modulus = 0.8

def rhs_rest_shape(v): # no rest shape relaxation, since we have no passive cells
    return 0

# package all into a single dict to pass to the optimizer method
energy_args = {'mod_bulk': mod_bulk, 'mod_shear': mod_shear,
               'angle_penalty': angle_penalty, 'bdry_penalty': bdry_penalty, 'epsilon_l': epsilon_l,
               'mod_area': 0, 'mod_perimeter': 0}
optimizer_args = {'energy_args': energy_args, 'tol': tol, 'maxiter': maxiter, 'verbose': True, 'bdr_weight': bdr_weight}

# directory for saving stuff
base_dir = "/data/Nikolas/GBE_simulation/runs/parameter_scans/"

###############################################
# define local function for running simulation
###############################################

def run_phase_diagm_sim(orientation, k_relax, k, noise_gaussian, feedback, noise_dynamic, use_voronoi, k_cutoff,
                        n_steps, random_seed=None, run_id=0):
    """
    Local function to run simulation in parallel.
    
    run_id can be used to additionally label the folder for saving, in case multiple simulations are run
    for the same parameter values.
    """
    
    # define directiory for saving stuff, save "metadata"
    save_dir = f"{base_dir}/{run_id}/orientation_{orientation}_kRelax_{k_relax}_kPassive_{k}_noiseGaussian_{noise_gaussian}_feedback_{feedback}_noiseDynamic_{noise_dynamic}_voronoi_{use_voronoi}_cutoff_{k_cutoff}_randomSeed_{random_seed}"
    try:
        os.makedirs(save_dir, exist_ok=True)
    except FileExistsError:
        print('Warning: directory exists')
    
    if "last_flipped_edges.pickle" in os.listdir(save_dir):
        return None # already done
        
    drs.save_self(save_dir, fname=None)
    logfile = save_dir + "/" + time_module.strftime("%Y-%m-%d_%H%M") + "log.txt"
    open(logfile, 'a').close()
    
    # set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # create initial condition
    print("")
    if orientation == "parallel":
        mesh_initial, property_dict = drs.create_rect_initial(n_y, n_x, noise=noise_gaussian, initial_strain=initial_strain,
                                                              orientation='parallel', isogonal=0, random_seed=random_seed,
                                                              boundaries=None, w_passive=0, w_passive_lr=0)
    elif orientation == "orthogonal":
        mesh_initial, property_dict = drs.create_rect_initial(n_x, n_y, noise=noise_gaussian, initial_strain=initial_strain,
                                                              orientation='orthogonal', isogonal=0, random_seed=random_seed,
                                                              boundaries=None, w_passive=0, w_passive_lr=0)
    
    with open(logfile, 'a') as f:
        f.write(f"{-1}\tGenerated initial condition\n")
    with open(f'{save_dir}/property_dict.pickle', 'wb') as handle:
        pickle.dump(property_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    passive_ids, passive_cells = (property_dict['passive_faces'], property_dict['passive_cells'])
    cell_id_to_modulus = np.vectorize(lambda x: 1-rel_elastic_modulus*(x in passive_cells))
    optimizer_args['cell_id_to_modulus'] = cell_id_to_modulus
    
    def params_pattern(fid):
        if fid in passive_ids:
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
        if noise_dynamic == 0:
            if feedback == "perimeter":
                mesh.euler_step(dt=dt, rhs_tension=tns.excitable_dt_act_pass_new, params=params_pattern,
                                rhs_rest_shape=rhs_rest_shape)
            elif feedback == "area":
                mesh.euler_step(dt=dt, rhs_tension=tns.excitable_dt_act_pass_new_area, params=params_pattern,
                    rhs_rest_shape=rhs_rest_shape)
        elif noise_dynamic > 0:
            if feedback == "perimeter":
                mesh.euler_maruyama_step(dt=dt, rhs_tension=tns.excitable_dt_act_pass_new, params=params_pattern,
                                         sigma=noise_dynamic, rhs_rest_shape=rhs_rest_shape)
            elif feedback == "area":
                mesh.euler_maruyama_step(dt=dt, rhs_tension=tns.excitable_dt_act_pass_new_area, params=params_pattern,
                                         sigma=noise_dynamic, hs_rest_shape=rhs_rest_shape)

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
        if not use_voronoi:
            sol = mesh.optimize_cell_shape(**optimizer_args, return_sol=True)
            with open(logfile, 'a') as f:
                f.write(f"{i}\tCell shape optimization: {sol.message}\n")
            if sol["status"] !=0 and (sol["message"] == "NaN result encountered."):
                break
        else:
            mesh.set_voronoi()
        # check for intercalation
        flipped, failed_flip = mesh.intercalate(exclude=list(msh.flatten(last_flipped_edges[-forbid_reflip:])),
                                                minimal_l=minimal_l, reoptimize=False,
                                                optimizer_args=optimizer_args)
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
    return None

###############################################
# define simulation sweep
###############################################

orientations = ["parallel", "orthogonal"]
k_relaxs = [2, 1/4, 1/20] # 
k_passives = [4, 6, 12]
noise_gaussians = [0.05, 0.1, 0.15]
noise_dynamics = [0, 0.2]
feedbacks = ["perimeter", "area"]
use_voronois = [False, True]
k_cutoffs = [0.2, 0.3, 0.4] # 0.2 we alreay have from last simulation 

reps = np.array([1, 2, 3]) # number of replicates # 1, 2, 3

# k_relax and k passive dependence
to_do = list(itertools.product(orientations, k_relaxs, k_passives, [0.1,], ["perimeter",], [0,], [False,], [0.3,], reps))
# gaussian noise influence
to_do = to_do + list(itertools.product(orientations, [2,], [6,], [0.05, 0.15], ["perimeter",], [0,], [False,], [0.3,], reps))
# area vs perimeter
to_do = to_do + list(itertools.product(orientations, [2,], [6,], [0.1,], ["area",], [0,], [False,], [0.3,], reps))
# noise in tension dynamics
to_do = to_do + list(itertools.product(orientations, [2,], [6,], [0.1,], ["perimeter",], [0.2,], [False,], [0.3,], reps))
# Voronoi simulations
to_do = to_do + list(itertools.product(orientations, [2,], [6,], [0.1,], ["perimeter",], [0,], [True,], [0.3,], reps))
# cutoff dependence
to_do = to_do + list(itertools.product(orientations, [2], [6], [0.1,], ["perimeter",], [0,], [False,], [0.2, 0.4], reps))

# only noise in tension dynamics to be redone
to_do = list(itertools.product(orientations, [2,], [6,], [0.1,], ["perimeter",], [0.3, 0.4, 0.5], [False,], [0.3,], reps))


n_steps = 501
run_id = "parameter_scan_SI_once_more_again"

###############################################
# run simulation
###############################################

Parallel(n_jobs=12, prefer=None)(delayed(run_phase_diagm_sim)( 
    orientation, k_relax, k, noise_gaussian, feedback, noise_dynamic, use_voronoi, k_cutoff,
    n_steps=n_steps, random_seed=rep, run_id=run_id)
    for orientation, k_relax, k, noise_gaussian, feedback, noise_dynamic, use_voronoi, k_cutoff, rep in to_do)



