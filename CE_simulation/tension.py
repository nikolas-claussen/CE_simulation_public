# AUTOGENERATED! DO NOT EDIT! File to edit: ../01_tension_time_evolution.ipynb.

# %% auto 0
__all__ = ['get_E_dual_jac', 'vectors_angle', 'sides_area', 'sides_circum', 'sides_angles', 'angles_shape', 'sides_area_jac',
           'excitable_dt', 'TensionHalfEdge', 'TensionHalfEdgeMesh', 'polygon_area', 'polygon_perimeter', 'get_E_dual',
           'excitable_dt_act_pass']

# %% ../01_tension_time_evolution.ipynb 3
import CE_simulation.mesh as msh

# %% ../01_tension_time_evolution.ipynb 4
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.integrate import solve_ivp
from scipy import linalg, optimize

from tqdm.notebook import tqdm

import sys
from copy import deepcopy, copy
import pickle

# %% ../01_tension_time_evolution.ipynb 5
from dataclasses import dataclass
from typing import Union, Dict, List, Tuple, Iterable, Any
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

import functools

# %% ../01_tension_time_evolution.ipynb 7
# basic formulas for triangles

def vectors_angle(a: NDArray[Shape["*"],Float], b: NDArray[Shape["*"],Float]) -> float:
    """Angle between two vectors"""
    inner = (a@b)/np.sqrt((a@a)*(b@b))
    return np.arccos(inner)

def sides_area(Ts: NDArray[Shape["3,..."],Float]):
    """Triangle area from side lengths."""
    Ts = np.sort(Ts, axis=0)[::-1]
    A = (Ts[0]+(Ts[1]+Ts[2]))*(Ts[2]-(Ts[0]-Ts[1]))*(Ts[2]+(Ts[0]-Ts[1]))*(Ts[0]+(Ts[1]-Ts[2]))/16
    return np.sqrt(np.clip(A, 0, np.inf))

def sides_circum(Ts: NDArray[Shape["3,..."],Float]):
    """Triangle circumcircle from side lengths"""
    R = np.prod(Ts, axis=0)/(4*sides_area(Ts))
    return R

def sides_angles(Ts: NDArray[Shape["3,..."],Float]):
    """Triangle angles from side lengths. Sorted so that angle [i] is opposite to Ts[i]"""
    R = sides_circum(Ts)
    inds = np.argmax(Ts, axis=0)
    # need to take the smaller two angles so as to avoid error in arcsin for angles >90
    phis = np.arcsin(Ts/(2*R))
    if isinstance(inds, np.ndarray):
        phis[inds, np.arange(len(inds))] = np.pi-(phis.sum(axis=0)-phis[inds, np.arange(len(inds))])
    else:
        phis[inds] = np.pi-(phis.sum(axis=0)-phis[inds])
    return phis

def angles_shape(phis: NDArray[Shape["3"],Float]) -> float:
    """Shape order parameter from angles"""
    alpha, beta, gamma = phis
    x = np.sqrt(3)*np.sin(alpha)*np.sin(alpha+2*beta)
    y = (np.cos(alpha)*np.cos(alpha+2*beta) - np.cos(2*alpha))
    psi = np.arctan2(x, y) + np.pi

    Psi = 6+2*(np.cos(4*phis)-np.cos(2*phis)
               -np.cos(2*(phis-np.roll(phis, 1, axis=0)))).sum(axis=0)
    Psi /= (3-np.cos(2*phis).sum(axis=0))**2
    Psi = np.sqrt(Psi)
    
    psi_tilde = np.pi - np.abs((3*psi) % (2*np.pi) - np.pi)
    
    return np.array([psi_tilde, Psi])

# %% ../01_tension_time_evolution.ipynb 11
def sides_area_jac(Ts: NDArray[Shape["3"],Float]) -> float:
    """get jacobian of area change in edge length"""
    dA = np.array([0., 0., 0.])
    dA += np.array([1, 1, 1])   * (Ts[2]-(Ts[0]-Ts[1])) * (Ts[2]+(Ts[0]-Ts[1])) * (Ts[0]+(Ts[1]-Ts[2]))
    dA += (Ts[0]+(Ts[1]+Ts[2])) * np.array([-1, 1, 1])  * (Ts[2]+(Ts[0]-Ts[1])) * (Ts[0]+(Ts[1]-Ts[2]))
    dA += (Ts[0]+(Ts[1]+Ts[2])) * (Ts[2]-(Ts[0]-Ts[1])) * np.array([1, -1, 1])  * (Ts[0]+(Ts[1]-Ts[2]))
    dA += (Ts[0]+(Ts[1]+Ts[2])) * (Ts[2]-(Ts[0]-Ts[1])) * (Ts[2]+(Ts[0]-Ts[1])) * np.array([1, 1, -1])

    dA /= 32*(sides_area(Ts)+1e-5)  # I think it should be 32 not 48.
    return dA

# %% ../01_tension_time_evolution.ipynb 13
# tension time evolution in triangle with constrained area
# perimeter and circumcircle constraints work poorly

def excitable_dt(Ts: NDArray[Shape["3"],Float], m=2) -> float:
    """
    Time derivative of tensions under excitable tension model with constrained area.
    
    Implements d_dt T = T^m
    """
    dT_dt = Ts**m
    area_jac = sides_area_jac(Ts)
    area_jac /= np.linalg.norm(area_jac)
    dT_dt -= area_jac * (area_jac@dT_dt)
    return dT_dt

# %% ../01_tension_time_evolution.ipynb 22
@dataclass
class TensionHalfEdge(msh.HalfEdge):
    """Half edge with attributes storing total and active tension.""" 
    rest: Union[float, None] = None
    passive: float = 0.0

    def __repr__(self):
        repr_str = super().__repr__()
        repr_str = repr_str.replace('HalfEdge', 'TensionHalfEdge')
        if self.rest is not None and self.passive is not None:
            repr_str += f", rest={round(self.rest, ndigits=1)}, passive={round(self.passive, ndigits=1)}"
        return repr_str
    
    def unwrap(self, in_place=True):
        """Cast to HalfEdge base class."""
        he = self if in_place else copy(self)
        he.__class__ = msh.HalfEdge
        del he.rest
        del he.passive
        if not in_place:
            return the
        
    
@patch
def wrap_as_TensionHalfEdge(self: msh.HalfEdge, rest=None, passive=0.0, in_place=True
                           ) -> Union[None, TensionHalfEdge]:
    """
    In-place/copy upcast from HalfEdge to TensionHalfEdge.
    
    Internal, for use in TensionHalfEdgeMesh.__init__.
    """
    the = self if in_place else copy(self)
    the.__class__ = TensionHalfEdge
    the.rest = rest
    the.passive = passive
    if not in_place:
        return the

# %% ../01_tension_time_evolution.ipynb 24
class TensionHalfEdgeMesh(msh.HalfEdgeMesh):
    """
    HalfEdgeMesh with methods for active triangulation dynamics.
    
    Can be instantiated from a HalfEdgeMesh, or from the more basal ListOfVerticesAndFaces.
    """
    def __new__(cls, mesh: Union[msh.HalfEdgeMesh, None]= None):
        if isinstance(mesh, msh.HalfEdgeMesh):
            mesh = deepcopy(mesh)
            mesh.__class__ = cls
            return mesh
        else:
            return super().__new__(cls)
    def __init__(self, mesh: Union[msh.ListOfVerticesAndFaces, msh.HalfEdgeMesh]):
        if isinstance(mesh, msh.ListOfVerticesAndFaces):
            super().__init__(mesh)
        for he in self.hes.values():
            he.wrap_as_TensionHalfEdge(in_place=True)
            
    def save_mesh(self, fname: str, save_rest_passive=False) -> None:
        super().save_mesh(fname)
        if save_rest_passive:
            pickle.dump({key: val.rest for key, val in self.hes.items()}, open(f"{fname}_rest.p", "wb"))
            pickle.dump({key: val.passive for key, val in self.hes.items()}, open(f"{fname}_passive.p", "wb"))
    save_mesh.__doc__ = (msh.HalfEdgeMesh.save_mesh.__doc__
                         +'\n Can also pickle passive&rest attributes as dicts at fname_{rest/passive}.p')
    
    def unwrap(self):
        """In-place cast to HalfEdgeMesh base class."""
        self.__class__ = msh.HalfEdgeMesh
        for he in self.hes.values():
            he.unwrap(in_place=True)
    
    @staticmethod
    def load_mesh(fname: str, load_rest_passive=False):
        """
        Load from file as saved by mesh.save_mesh.
        
        Can load rest/passive attributes of half edges, if saved as pickled dicts at fname_{rest/passive}.p
        """
        mesh = TensionHalfEdgeMesh(super(TensionHalfEdgeMesh, TensionHalfEdgeMesh).load_mesh(fname))
        if load_rest_passive:
            rest_dict = pickle.load(open(f'{fname}_rest.p', 'rb'))
            passive_dict = pickle.load(open(f'{fname}_passive.p', 'rb'))
            for key, he in mesh.hes.items():
                he.rest = rest_dict[key]
                he.passive = passive_dict[key]
        return mesh

# %% ../01_tension_time_evolution.ipynb 30
@patch
def set_rest_lengths(self: TensionHalfEdgeMesh) -> None:
    """Set the triangulation rest lengths to current lengths"""
    for he in self.hes.values():
        he.rest = np.linalg.norm(he.vertices[1].coords - he.vertices[0].coords)

# %% ../01_tension_time_evolution.ipynb 38
@patch
def vertices_to_vector(self: TensionHalfEdgeMesh, flattened=True) -> NDArray[Shape["*"],Float]:
    """
    Format vertex coordinates for use in energy minimization.  
    
    Returns a vector of vertex coordinates. Vector entries are ordered by vertex keys, i.e.
    entry 0 corresponds to the coordinates of the vertex with lowest key.

    If flattened, flatten the vector into a 1d array according to C-ordering.
    Else, return n_vertices, 2 array.
    """
    vertex_keys = sorted(self.vertices.keys())
    vertex_vector = np.stack([self.vertices[key].coords for key in vertex_keys])
    if flattened:
        return vertex_vector.reshape(2*vertex_vector.shape[0])
    return vertex_vector
       
@patch
def vector_to_vertices(self: TensionHalfEdgeMesh, x0, flattened=True) -> Dict[int, NDArray[Shape["2"],Float]]:
    """Reverse of vertices_to_vector - format output of energy minimization as dict."""
    vertex_keys = sorted(self.vertices.keys())
    if flattened:
        vertex_vector = x0.reshape((int(x0.shape[0]/2), 2))
    else:
        vertex_vector = x0
    return {key: val for key, val in zip(vertex_keys, vertex_vector)}

# %% ../01_tension_time_evolution.ipynb 40
import jax.numpy as jnp
from jax import grad as jgrad
from jax import jit
from jax.nn import relu as jrelu
from jax.tree_util import Partial
from jax.config import config

config.update("jax_enable_x64", True) 

# %% ../01_tension_time_evolution.ipynb 41
@jit
def polygon_area(pts: NDArray[Shape["*,2,..."], Float]) -> NDArray[Shape["2,..."], Float]:
    """JAX-compatible - area of polygon. Assuming no self-intersection. Pts.shape (n_vertices, 2)"""
    return jnp.sum(pts[:,0]*jnp.roll(pts[:,1], 1, axis=0) - jnp.roll(pts[:,0], 1, axis=0)*pts[:,1], axis=0)/2

@jit
def polygon_perimeter(pts: NDArray[Shape["*,2,..."], Float], epsilon_l=1e-4) -> NDArray[Shape["2,..."], Float]:
    """
    JAX-compatible - perimeter of polygon. Assuming no self-intersection. pts.shape (n_vertices, 2).
    
    `epsilon_l` is a mollifiert which ensures differentiability at short lengths 
    """
    
    return jnp.sum(jnp.sqrt(jnp.sum((pts-jnp.roll(pts, 1, axis=0))**2, axis=1)+epsilon_l), axis=0)

# %% ../01_tension_time_evolution.ipynb 44
@patch
def serialize_triangulation(self: TensionHalfEdgeMesh):
    """
    Serialize triangulation structure into numpy arrays.
    
    "Serializes" a HalfEdgeMesh into a bunch of arrays which can be used by a JAX-compatible
    function to compute the "elastic energy" used to flatten a triangulation.
    
    Returns
    -------
    
    Dict with the following entries
    
    e_lst : (n_edges, 2) array
        Indices to the serialized vertex positions defining edges in triangulation
    rest_lengths : (n_edges) array
        Rest lengths of each edge
    tri_lst : (n_triangles, 3) array
        Indices to the serialized vertex positions defining tr iangles in triangulation
    """
    e_lst = []
    tri_lst = []
    rest_lengths = []

    # we will need to look up which vertex key corresponds to list position
    vertex_key_dict = {key: ix for ix, key in enumerate(sorted(self.vertices.keys()))}
    
    for e in self.hes.values():
        if e.duplicate: # avoid duplicates
            e_lst.append([vertex_key_dict[v._vid] for v in e.vertices])
            rest_lengths.append((e.rest+e.twin.rest)/2)
    e_lst = jnp.array(e_lst).T
    rest_lengths = jnp.array(rest_lengths)
    
    for fc in self.faces.values():
        tri_lst.append([vertex_key_dict[v._vid] for v in fc.vertices][::-1])
    tri_lst = jnp.array(tri_lst).T
    n_vertices = len(self.vertices)
    
    return {'e_lst': e_lst, 'rest_lengths': rest_lengths, 'tri_lst': tri_lst}

# %% ../01_tension_time_evolution.ipynb 45
@jit
def get_E_dual(x0: NDArray[Shape["*"],Float], e_lst: NDArray[Shape["*, 2"],Float],
               rest_lengths: NDArray[Shape["*"],Float], tri_lst: NDArray[Shape["*, 3"],Float],
               mod_area=0.01, A0=jnp.sqrt(3)/4) -> float:
    """
    Dual energy function for triangulation flattening
    
    Compute the deviation of a set of vertex positions from the desired rest lengths 
    sum_e (|x_i-x_j| - l_e)^2, plus regularization term for triangle areas.
    Uses x0, the serialized vector of vertex positions, and the arrays returned by
    HalfEdgeMesh.get_dual_energy_fct_jax.
    
    Performance note: calling this function for meshes with different numbers of
    edges & vertices will trigger JIT-recompilation which can be slow. Calling on meshes with the
    same number of edges & vertices (e.g. related by T1s) is OK.

    Parameters
    ----------
    x0: (2*n_vertices,) array
        Vertex position vector as created by HalfEdgeMesh.vector_to_vertices.
    e_lst, rest_lengths, tri_lst: arrays
        Created by HalfEdgeMesh.get_dual_energy_fct_jax
    mod_area: float
        Strength of area regularization
    A0: float
        Triangle reference area for area regularization 
    
    Returns
    -------
    float
        elastic energy
    
    """
    pts = x0.reshape((int(x0.shape[0]/2), 2))
    
    lengths = jnp.linalg.norm(pts[e_lst[0]]-pts[e_lst[1]], axis=1)
    E_length = 1/2 * jnp.sum((lengths-rest_lengths)**2)
    # triangle area penalty
    A = polygon_area(pts[tri_lst].transpose((0,2,1)))
    # orientation penalty:
    E_area = mod_area/2 *(100*jnp.sum(jrelu(-A+A0/4)**2) + jnp.sum((A-A0)**2))
    # relu term penalizes 'flipped' triangles with incorrect orientation.
    
    return E_length + E_area
  
get_E_dual_jac = jit(jgrad(get_E_dual))

# %% ../01_tension_time_evolution.ipynb 52
@patch
def flatten_triangulation(self: TensionHalfEdgeMesh, tol=1e-4, verbose=True, mod_area=0.01, A0=jnp.sqrt(3)/4,
                          reset_intrinsic=True, return_sol=False) -> Union[None, Dict]:
    """
    Flatten triangulation - optimize vertex positions to match intrinsic lengths.
    
    This wrapper does the following:
    1) Serialize a HalfEdgeMesh into arrays using HalfEdgeMesh.get_dual_energy_fct_jax
    and HalfEdgeMesh.vector_to_vertices
    2) Optimize the dual energy function get_E_dual using conjugate gradient implemented
    by scipy.optimize.minimize
    3) De-serialize the result and update the vertex positions
    4) Optionally, re-set the intrinsic lengths to match the phyiscal edge lengths
    
    Performance note: calling this function for meshes with different numbers of
    edges & vertices will trigger JIT-recompilation which can be slow. Calling on meshes with the
    same number of edges & vertices (e.g. related by T1s) is OK.
    
    Parameters
    ----------
    tol : float
        Optimizer tolerance
    verbose : bool
        Print warnings if optimization fails
    mod_area : float
        Area regularization, passed on to get_E_dual
    A0 : float
        Reference triangle area, passed on to get_E_dual
    reset_intrinsic : bool
        Reset intrinsic lengths after optimization
    return_sol : bool
        Return optimizer result dict
    """
    energy_arrays = self.serialize_triangulation()
    args = tuple([energy_arrays[key] for key in ['e_lst', 'rest_lengths', 'tri_lst']]+[mod_area, A0])
    x0 = self.vertices_to_vector()
    sol = optimize.minimize(get_E_dual, x0, method="CG", jac=get_E_dual_jac, tol=tol, args=args)
    sol['initial_fun'] = float(get_E_dual(x0, *args))
    if sol["status"] !=0 and verbose:
        print("Triangulation optimization failed")
        print(sol["message"])
    new_coord_dict = self.vector_to_vertices(sol["x"])
    for key, val in self.vertices.items():
        val.coords = new_coord_dict[key]
    if reset_intrinsic:
        self.set_rest_lengths()
    if return_sol:
        return sol

# %% ../01_tension_time_evolution.ipynb 70
@patch
def get_angles(self: TensionHalfEdgeMesh) -> Dict[int, float]:
    """Get the angle opposite to each half edge in the mesh."""
    angle_dict = {}
    egde_lengths = self.get_edge_lens()
    for fc in self.faces.values():
        heids = [he._heid for he in fc.hes]
        angles = sides_angles([egde_lengths[e] for e in heids]) 
        for e, a in zip(heids, angles):
            angle_dict[e] = a   
    return angle_dict

@patch
def get_double_angles(self: TensionHalfEdgeMesh) -> Dict[int, float]:
    """Get the sum of the opposite angles of an edge (2x half edge) in mesh, e.g. for Delaunay criterion."""
    angles = self.get_angles()
    double_angles = {he._heid: (angles[he._heid]+angles[he._twinid]) for he in self.hes.values()
                             if (he.face is not None) and (he.twin.face is not None)}
    return double_angles

# %% ../01_tension_time_evolution.ipynb 73
def excitable_dt_act_pass(Ts: NDArray[Shape["3"], Float], Tps: NDArray[Shape["3"], Float], k=1, m=2, k_cutoff=0,
                          ) -> Tuple[NDArray[Shape["3"],Float],NDArray[Shape["3"],Float]]:
    """
    Time derivative of tensions under excitable tension model, including passive tension.
    
    Implements the following equations:
        d_dt T = T^m
        d_dt T_passive = -k*T_passive
    
    with the following additions:
        - a -k_cutoff*T^(m+1) term which cuts of excitable feedback at large tensions for numerical stability
        - projection of the d_dt T - vector on triangle-area-preserving edge length changes
    
    For m==1 (no excitable tension feedback), we implement a special case:
        d_dt T = -k*(T-1)
    i.e. tensions relax back to equilateral. This will be useful later to model completely
    passive edges with no excitable dynamics.
    
    Parameters
    ----------
    Ts : (3,) array
        active tensions
    Tps : (3,) array
        passive tensions
    k : float
        passive tension relaxation rate
    m : float
        excitable tension exponent
    k_cutoff : 
        cutoff for excitable tension. 0 = no cutoff.
        
    Returns
    -------
    dT_dt : (3,) array
        time derivative of active tension
    dTp_dt : (3,) array
        time derivative of passive tension


    """
    dT_dt = (m!=1)*((Ts-Tps)**m - k_cutoff*(Ts-Tps)**(m+1) - k*Tps) - k*(m==1)*(Ts-1)    
    dTp_dt = -k*Tps
    area_jac = sides_area_jac(Ts-Tps)
    area_jac /= np.linalg.norm(area_jac)
    dT_dt -= area_jac * (area_jac@dT_dt)    
    return dT_dt, dTp_dt

# %% ../01_tension_time_evolution.ipynb 75
@patch
def reset_rest_passive_flip(self: TensionHalfEdgeMesh, e: TensionHalfEdge, method="smooth") -> None:
    """
    Reset rest length and passive tensions of flipped he according to myosin handover.
    
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

# %% ../01_tension_time_evolution.ipynb 76
@patch
def euler_step(self: TensionHalfEdgeMesh, dt=.005, rhs_tension=excitable_dt_act_pass, params=None,
               rhs_rest_shape: Union[None, callable]=None) -> None:
    """
    Euler step intrinsic edge length and reference shapes. 
    
    Iterates over mesh triangles and cells, updating the intrinsic properties (active and passive tensions,
    reference shapes) using the provided ODE RHS functions.
    
    Implements spatial patterning of the tension evolution equations via the params keyword.
    
    Parameters
    ----------
    dt : float
        Time step
    rhs_tension : callable
        Function which takes arguments (T_active, T_passive) and returns their time derivatives
    params : dict or callable
        Parameters for the rhs_tension function. Can be a function from face ids -> parameter dict,
        allowing different triangles to evolve differently.
    rhs_rest_shape : callable
        Function which takes a vertex as argument and returns the time derivative of the rest shape
        (e.g. viscous relaxation). If None, the function returns 0.
    
    """
    # Euler step edges
    for fc in self.faces.values():
        # collect edges
        Ts, Tps = (np.array([he.rest for he in fc.hes]), np.array([he.passive for he in fc.hes]))
        if isinstance(params, dict):
            dT_dt, dTp_dt = rhs_tension(Ts, Tps, **params)
        elif callable(params):
            dT_dt, dTp_dt = rhs_tension(Ts, Tps, **params(fc._fid))
        Ts += dt*dT_dt
        Tps += dt*dTp_dt
        for T, Tp, he in zip(Ts, Tps, fc.hes):
            he.rest = T
            he.passive = Tp
    # Euler step cells, if desired
    if rhs_rest_shape is not None:
        for v in self.vertices.values():
            v.rest_shape += dt*rhs_rest_shape(v)
