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
import itertools

from joblib import Parallel, delayed

import jax
import jax.numpy as jnp
from jax import jit
from jax.config import config

jax.default_device(jax.devices('cpu')[0])
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)

###############################################
# define local functions if necessary
###############################################



###############################################
# set parameters
###############################################

# initial condition parameters

n_x = 70
n_y = 70
chain_time = 600
n_chains = 1000

# directory for saving stuff

base_dir = "/data/Nikolas/GBE_simulation/runs/hard_disks"
run_id = "hard_disk_ltc_samples"


###############################################
# define local function for running simulation
###############################################

def get_hard_disk(eta, random_seed=None):
    """
    Local function to run simulation in parallel.
    
    run_id can be used to additionally label the folder for saving, in case multiple simulations are run
    for the same parameter values.
    """
    # define directiory for saving stuff, save "metadata"
    save_name = f"{base_dir}/{run_id}/eta_{eta}_randomSeed_{random_seed}_mesh"
    
    # set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # create initial condition
    mesh_initial, _ = ecm.create_hard_disk_initial(n_x=n_x, n_y=n_y, eta=eta,
                                                   initial_strain=0, noise_gaussian=0, remove_boundary_dx=1,
                                                   n_chains=n_chains, chain_time=chain_time, progress_bar=False)
    
    mesh_initial.save_mesh(save_name, save_attribs=False)

    return None


###############################################
# define simulation sweep
###############################################

etas = np.array([0.1, 0.15, 0.2, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
random_seeds = [1, 2, 3]

to_do = itertools.product(etas, random_seeds)
drs.save_self(f"{base_dir}/{run_id}", fname=None)

###############################################
# run simulation
###############################################

Parallel(n_jobs=32, prefer=None)(delayed(get_hard_disk)(eta=eta, random_seed=random_seed)
                                 for eta, random_seed in to_do)





