# AUTOGENERATED! DO NOT EDIT! File to edit: ../01_tension_time_evolution.ipynb.

# %% auto 0
__all__ = ['vectors_angle', 'sides_area', 'sides_circum', 'sides_angles', 'angles_shape', 'sides_area_jac', 'excitable_dt',
           'excitable_dt_post']

# %% ../01_tension_time_evolution.ipynb 3
from .triangle import *

# %% ../01_tension_time_evolution.ipynb 4
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from numpy import sin, cos, tan, pi, sqrt, arccos, arctan, arctan2
from numpy.linalg import norm

from scipy.integrate import solve_ivp
from scipy import ndimage
from scipy import spatial
from scipy import optimize

from tqdm.notebook import tqdm

import sys
from copy import deepcopy

# %% ../01_tension_time_evolution.ipynb 5
from dataclasses import dataclass
from typing import Union, Dict, List, Tuple, Iterable
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../01_tension_time_evolution.ipynb 7
# basic formulas for triangles

def vectors_angle(a,b):
    """Angle between two vectors"""
    inner = (a@b)/sqrt((a@a)*(b@b))
    return arccos(inner)

def sides_area(Ts):
    """Triangle area from side lengths"""
    Ts = np.sort(Ts, axis=0)[::-1]
    #A = sqrt((Ts[0]+Ts[1]+Ts[2])*(-Ts[0]+Ts[1]+Ts[2])*(Ts[0]-Ts[1]+Ts[2])*(Ts[0]+Ts[1]-Ts[2]))/4
    A = (Ts[0]+(Ts[1]+Ts[2]))*(Ts[2]-(Ts[0]-Ts[1]))*(Ts[2]+(Ts[0]-Ts[1]))*(Ts[0]+(Ts[1]-Ts[2]))/16
    return sqrt(np.clip(A, 0, np.inf))

def sides_circum(Ts):
    """Triangle circumcircle from side lengths"""
    R = np.prod(Ts, axis=0)/(4*sides_area(Ts))
    return R

def sides_angles(Ts):
    """Triangle angles from side lengths. Sorted so that angle [i] is opposite to Ts[i]"""
    R = sides_circum(Ts)
    inds = np.argmax(Ts, axis=0)
    # need to take the smaller two angles so as to avoid error in arcsin for angles >90
    phis = np.arcsin(Ts/(2*R))
    if isinstance(inds, np.ndarray):
        phis[inds, np.arange(len(inds))] = pi-(phis.sum(axis=0)-phis[inds, np.arange(len(inds))])
    else:
        phis[inds] =  pi-(phis.sum(axis=0)-phis[inds])
    return phis

def angles_shape(phis):
    """Shape order parameter from angles"""
    alpha, beta, gamma = phis
    x = sqrt(3)*sin(alpha)*sin(alpha+2*beta)
    y = (cos(alpha)*cos(alpha+2*beta) - cos(2*alpha))
    psi = np.arctan2(x, y) + pi

    Psi = 6+2*(cos(4*triangle)-cos(2*triangle) - cos(2*(triangle-np.roll(triangle, 1, axis=0)))).sum(axis=0)
    Psi /= (3-cos(2*triangle).sum(axis=0))**2
    Psi = np.sqrt(Psi)
    
    psi_tilde = pi - np.abs((3*psi) % (2*pi) - pi)
    
    return np.array([psi_tilde, Psi])

# %% ../01_tension_time_evolution.ipynb 11
def sides_area_jac(Ts):
    """get jacobian of area change in edge length"""
    dA = np.array([0., 0., 0.])
    dA += np.array([1, 1, 1])   * (Ts[2]-(Ts[0]-Ts[1])) * (Ts[2]+(Ts[0]-Ts[1])) * (Ts[0]+(Ts[1]-Ts[2]))
    dA += (Ts[0]+(Ts[1]+Ts[2])) * np.array([-1, 1, 1])  * (Ts[2]+(Ts[0]-Ts[1])) * (Ts[0]+(Ts[1]-Ts[2]))
    dA += (Ts[0]+(Ts[1]+Ts[2])) * (Ts[2]-(Ts[0]-Ts[1])) * np.array([1, -1, 1])  * (Ts[0]+(Ts[1]-Ts[2]))
    dA += (Ts[0]+(Ts[1]+Ts[2])) * (Ts[2]-(Ts[0]-Ts[1])) * (Ts[2]+(Ts[0]-Ts[1])) * np.array([1, 1, -1])

    dA /= 48*(sides_area(Ts)+1e-5)
    return dA

# %% ../01_tension_time_evolution.ipynb 13
# tension time evolution in triangle with constrained area
# perimeter and circumcircle constraints work poorly

def excitable_dt(Ts, m=2):
    """Time derivative of tensions under excitable tension model with constrained area"""
    dT_dt = Ts**m
    area_jac = sides_area_jac(Ts)
    area_jac /= norm(area_jac)
    dT_dt -= area_jac * (area_jac@dT_dt)
    return dT_dt

# %% ../01_tension_time_evolution.ipynb 24
@patch
def get_angles(self: HalfEdgeMesh):
    angle_dict = {}
    egde_lengths = self.get_edge_lens()
    for fc in self.faces.values():
        lengths = []
        heids = []
        returned = False
        start_he = fc.hes[0]
        he = start_he
        while not returned:
            heids.append(he._heid)
            lengths.append(egde_lengths[he._heid])
            he = he.nxt
            returned = (he == start_he)
        angles = sides_angles(lengths) 
        for heid, a in zip(heids, angles):
            angle_dict[heid] = a   
    return angle_dict

@patch
def get_double_angles(self: HalfEdgeMesh):
    angles = self.get_angles()
    double_angles = {he._heid: (angles[he._heid]+angles[he._twinid]) for he in self.hes.values()
                             if (he.face is not None) and (he.twin.face is not None)}
    return double_angles

# %% ../01_tension_time_evolution.ipynb 32
import autograd.numpy as anp  # Thinly-wrapped numpy
from autograd import grad as agrad

from scipy.sparse import csc_matrix

# %% ../01_tension_time_evolution.ipynb 33
@patch
def vertices_to_initial_cond(self: HalfEdgeMesh):
    """Format vertices for use in energy minimization."""
    vertex_keys = sorted(self.vertices.keys())
    vertex_vector = np.stack([self.vertices[key].coords for key in vertex_keys]).T
    return np.hstack([vertex_vector[0], vertex_vector[1]])
       
@patch
def initial_cond_to_vertices(self: HalfEdgeMesh, x0):
    """Reverse of format vertices for use in energy minimization."""
    vertex_keys = sorted(self.vertices.keys())
    x, y = (x0[:int(len(x0)/2)], x0[int(len(x0)/2):])
    vertex_vector = np.stack([x, y], axis=1)
    return {key: val for key, val in zip(vertex_keys, vertex_vector)}

@patch
def get_energy_fct(self: HalfEdgeMesh):
    """Get energy function sum_edges (l_e -l_e,0)^2. remove translation mode by keeping COM fixed."""
    e_lst = []
    rest_lengths = []

    # we will need to look up which vertex key corresponds to list position
    vertex_key_dict = {key: ix for ix, key in enumerate(sorted(self.vertices.keys()))}
    for e in self.hes.values():
        if e.duplicate: # avoid duplicates
            e_lst.append([vertex_key_dict[v._vid] for v in e.vertices])
            rest_lengths.append((e.rest+e.twin.rest)/2)

    e_lst = anp.array(e_lst).T
    rest_lengths = anp.array(rest_lengths)
    center = anp.mean([val.coords for val in self.vertices.values()], axis=0)
    n_vertices = len(self.vertices)
    def get_E(x0):
        x, y = (x0[:n_vertices], x0[n_vertices:])
        lengths = anp.sqrt((x[e_lst[0]]
                            -x[e_lst[1]])**2
                           + (y[e_lst[0]]-y[e_lst[1]])**2)
        E = 1/2 * anp.sum((lengths-rest_lengths)**2)
        # displacement from initial center
        E = E + 1/2*((anp.mean(x)-center[0])**2+(anp.mean(y)-center[0]))**2
        return E
    
    return get_E, agrad(get_E)

# %% ../01_tension_time_evolution.ipynb 66
def excitable_dt_post(Ts, Tps, k=1, m=2):
    """Time derivative of tensions under excitable tension model with constrained area,
    with passive tension for post intercalation"""
    dT_dt = (Ts-Tps)**m - k*Tps
    dTp_dt = -k*Tps
    area_jac = sides_area_jac(Ts-Tps)
    area_jac /= norm(area_jac)
    dT_dt -= area_jac * (area_jac@dT_dt)
    return dT_dt, dTp_dt

@patch
def reset_rest_passive_flip(self: HalfEdgeMesh, e: HalfEdge, method="smooth"):
    """Reset rest length and passive tensions of flipped he according to myosin inheritance.
    Two options: "smooth" results in contiuous rest lengths, using the passive contruction,
    "direct" directly sets the rest length to the values of the neighbors.
    """
    twin = e.twin
    rest_pre = (e.rest+twin.rest)/2
    rest_neighbors = (e.nxt.rest+e.prev.rest+twin.nxt.rest+twin.prev.rest)/4
    if method == "smooth":
        e.rest = np.linalg.norm(e.vertices[0].coords - e.vertices[1].coords)
        e.passive = (rest_pre+e.rest)-2*rest_neighbors
        twin.rest, twin.passive = (e.rest, e.passive)
    elif method == "direct":
        e.rest = rest_neighbors
        e.passive = rest_pre-rest_neighbors
        twin.rest, twin.passive = (e.rest, e.passive)
    else:
        print("method must be smooth or direct")
