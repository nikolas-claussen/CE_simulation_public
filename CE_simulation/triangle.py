# AUTOGENERATED! DO NOT EDIT! File to edit: ../00_triangle_data_structure.ipynb.

# %% auto 0
__all__ = ['removekey', 'flatten', 'sort_vertices', 'sort_ids_by_vertices', 'get_neighbors', 'ListOfVerticesAndFaces', 'HalfEdge',
           'Vertex', 'Face', 'Edge', 'get_half_edges', 'HalfEdgeMesh']

# %% ../00_triangle_data_structure.ipynb 3
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import spatial

# %% ../00_triangle_data_structure.ipynb 4
from collections import defaultdict

# %% ../00_triangle_data_structure.ipynb 5
from dataclasses import dataclass
from typing import Union, Dict, List, Tuple, Iterable
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../00_triangle_data_structure.ipynb 7
def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

# %% ../00_triangle_data_structure.ipynb 8
from bisect import bisect_left

# %% ../00_triangle_data_structure.ipynb 10
def flatten(lst, max_depth=1000, iter_count=0):
    """
    Flatten a list of lists into a list.

    Also works with inhomogeneous lists, e.g., [[0,1],2]. The argument
    depth determines how "deep" to flatten the list, e.g. with max_depth=1:
    [[(1,0), (1,0)]] -> [(1,0), (1,0)].

    Parameters
    ----------
    lst : list
        list-of-lists.
    max_depth : int, optional
        To what depth to flatten the list.
    iter_count : int, optional
        Helper argument for recursion depth determination.
    Returns
    -------
    iterator
        flattened list.

    """
    for el in lst:
        if (isinstance(el, Iterable) and not isinstance(el, (str, bytes))
                and iter_count < max_depth):
            yield from flatten(el, max_depth=max_depth,
                               iter_count=iter_count+1)
        else:
            yield el

# %% ../00_triangle_data_structure.ipynb 11
def sort_vertices(vertices: np.ndarray) -> np.ndarray:
    """Sort vertices of cycle counter clockwise by polar angle. Guaranteed to work for non-convex polygons."""
    vertices -= np.mean(vertices, axis=0) # center
    phis = [np.arctan2(*x[::-1]) for x in vertices]
    return np.stack([x for _, x in sorted(zip(phis, vertices))])

def sort_ids_by_vertices(ids: Iterable[int], vertices: Iterable[NDArray]) -> list:
    """Like sort_vertices, sort ids of cycle counter clockwise by polar angle."""
    vertices -= np.mean(vertices, axis=0) # vertices
    phis = [np.arctan2(*x[::-1]) for x in vertices]
    return [x for _, x in sorted(zip(phis, ids))]

# %% ../00_triangle_data_structure.ipynb 19
def get_neighbors(faces):
    """compute neighbor list by checking which triangles share 2 vertices."""
    edge_dict = defaultdict(list)
    for key, fc in faces.items():
        edges = [tuple(sorted((fc+[fc[0]])[i:i+2])) for i in range(len(fc))]
        [edge_dict[e].append(key) for e in edges]

    neighbors = defaultdict(list)
    for edge, fcs in edge_dict.items():
        if len(fcs) == 2:
            neighbors[fcs[0]].append(fcs[1])
            neighbors[fcs[1]].append(fcs[0])
    return neighbors

# %% ../00_triangle_data_structure.ipynb 20
class ListOfVerticesAndFaces:
    def __init__(self, vertices, faces, neighbors=None):
        # if we pass lists, automatically assign ids to triangles and vertices
        vertices = vertices if type(vertices) is dict else {ix: x for ix, x in enumerate(vertices)}
        faces = faces if type(faces) is dict else {ix: x for ix, x in enumerate(faces)}
        # ensure that all triangles are ordered cc-wise
        faces = {key: sort_ids_by_vertices(fc, [vertices[x] for x in fc])
                 for key, fc in faces.items()}
        self.vertices, self.faces = (vertices, faces)
        self.neighbors = get_neighbors(faces) if neighbors is None else neighbors
        
    # some basic methods
    def remove_vertex(self, v_id):
        del self.vertices[v_id]
        self.faces = {key: face for key, face in self.faces.items() if not v_id in face}
        self.neighbors = get_neighbors(self.faces)
        
    def get_combined_edges(self):
        """Get a list of unique edges. Edges are a tuple ((vert 1, vert 2), (face 1, face 2)). Boundary edges 
        have face 1 None."""
        combined_edges = []
        for fc_key, fc in self.faces.items():
            neighbors = self.neighbors[fc_key]
            vertex_edges = [(fc+[fc[0]])[i:i+2] for i in range(len(fc))]
            for vertex_edge in vertex_edges:
                # check if it is shared with neighbor
                other_key = [nghb for nghb in neighbors if all([v in self.faces[nghb] for v in vertex_edge])]
                face_edge = sorted([fc_key, other_key[0]]) if other_key != [] else (None, fc_key)
                combined_edge = (tuple(sorted(vertex_edge)), tuple(face_edge))
                combined_edges.append(combined_edge)
        return set(combined_edges)
        
    @staticmethod
    def fromObj(fname):
        """Read from .onj file. If {fname}_ids.txt is present, read ids from that."""
        with open(fname+'.obj') as f:
            lns = f.readlines()
            vertices = [np.array([float(x) for x in ln[2:-1].split(" ")])[:2] # remove z-coord
                        for ln in lns if ln.startswith("v")]
            faces = [[int(x)-1 for x in ln[2:-1].split(" ")] # to start counting from 0 again
                      for ln in lns if ln.startswith("f")]
        if os.path.isfile(fname+'_ids.txt'): # read ids if defined
            with open(fname+'_ids.txt') as f:
                lns = f.readlines()
                vertex_ids = [int(ln[2:-1]) for ln in lns if ln.startswith("v")]
                face_ids = [int(ln[2:-1]) for ln in lns if ln.startswith("f")]
            vertices = {v_id: v for v_id, v in zip(vertex_ids, vertices)}
            faces = {fcid: fc for fcid, fc in zip(face_ids, faces)}
        return ListOfVerticesAndFaces(vertices, faces)

# %% ../00_triangle_data_structure.ipynb 25
@patch
def saveObj(self:ListOfVerticesAndFaces, fname, save_ids=False):
    """save as obj file. .obj automatically appended to fname. If save_ids is True, also save a list
    of vertex and face ids."""
    # create a sorted list of vertices
    vertex_keys = sorted(self.vertices.keys())
    vertex_list = [self.vertices[key] for key in vertex_keys]
    # change faces list to refer to this ordered list. Counting from 1 for .obj
    face_keys = sorted(self.faces.keys())
    faces_list = [[bisect_left(vertex_keys, v)+1 for v in self.faces[key]] for key in face_keys]
    # overwrite
    try:
        os.remove(fname+".obj")
    except OSError:
        pass
    # write
    with open(fname+".obj", "a") as f:
        f.write('# vertices\n')
        for pt in vertex_list:
            to_write = ' '.join(['v'] + [str(x) for x in pt] + ['0']) + '\n'  # include z-ccoord
            f.write(to_write)
        f.write('# faces\n')
        for fc in faces_list:
            to_write = ' '.join(['f'] + [str(x) for x in fc]) + '\n'
            f.write(to_write)
    if save_ids:
        try:
            os.remove(fname+"_ids.txt")
        except OSError:
            pass
        with open(fname+"_ids.txt", "a") as f:
            f.write('# vertex IDs corresponding to .obj file\n')
            for key in vertex_keys:
                f.write('v '+str(key)+'\n')
            f.write('# face IDs corresponding to .obj file\n')
            for key in face_keys:
                f.write('f '+str(key)+'\n')


# %% ../00_triangle_data_structure.ipynb 33
from dataclasses import dataclass, field

# %% ../00_triangle_data_structure.ipynb 34
@dataclass
class HalfEdge:
    """Attribute holder class for half edges. Attributes point to other items."""
    _heid : int
    nxt: int
    prev: int
    twin: int
    face: Union[int, None] # None if it's a boundary
    vertices: tuple # 0 is origin, 1 is destination
    rest: float = 0.
    passive: float = 0.
    flipped: int = 0
    variables : dict = field(default_factory=dict) 
    # further variables for optimization. Maybe have rest, passive, flipped as true attribs?
    
@dataclass
class Vertex:
    """Attribute holder class for vertices. Attributes point to other items. Note: different from the
    standard half edge data structure, I store all incident he's, for latter convenience (e.g. force balance)
    computation."""
    _vid : int
    coords : NDArray[Shape["2"], Float]
    incident : List[HalfEdge]

@dataclass
class Face:
    """Attribute holder class for faces. Attributes point to other items."""
    _fid : int
    hes : List[HalfEdge]

# %% ../00_triangle_data_structure.ipynb 40
# obsololete?
@dataclass
class Edge:
    """Attribute holder class for edges. Main point is to use it to store variables for ODE evolution"""
    _eid : int
    hes : Tuple[HalfEdge, HalfEdge]
    variables : dict
        
    def __post_init__(self):
        assert (self.hes[0].twin == self.hes[1]._heid) and (self.hes[1].twin == self.hes[0]._heid)

# %% ../00_triangle_data_structure.ipynb 48
def get_half_edges(mesh: ListOfVerticesAndFaces) -> Dict[int, HalfEdge]:
    """Create list of half-edges from a ListOfVerticesAndFaces mesh"""
    heid_counter = 0
    he_vertex_dict = dict()
    # first create half edges without their twins by going around each face.
    # index them by their vertices to match twins after
    for key, fc in mesh.faces.items():
        # ensure face is oriented correctly
        fc = sort_ids_by_vertices(fc, [mesh.vertices[x] for x in fc])
        heids = [heid_counter+i for i in range(len(fc))]
        nxts, prevs = (np.roll(heids, +1).tolist(), np.roll(heids, -1).tolist())
        vertices = [tuple((fc+[fc[0]])[i:i+2]) for i in range(len(fc))]
        for _heid, nxt, prev, verts in zip(heids, nxts, prevs, vertices):
             he_vertex_dict[verts] = HalfEdge(_heid, prev, nxt, None, key, verts)
        heid_counter += len(fc)
    # now match the half-edges. if they cannot match, add a new he with faec None
    hes = []
    for he1 in he_vertex_dict.values():
        try:
            he2 = he_vertex_dict[he1.vertices[::-1]]
        except KeyError:
            he2 = HalfEdge(heid_counter, None, None, he1._heid, None, he1.vertices[::-1],)
            heid_counter += 1
        he1.twin, he2.twin = (he2._heid, he1._heid)
        hes.append(he1); hes.append(he2)
    # find the "next" of the boundary edges. we can just traverse inshallah
    bdry = [he for he in hes if he.face is None]
    for he1 in bdry:
        try:
            nxt = next(he2 for he2 in bdry if he1.vertices[1] == he2.vertices[0])
            prev = next(he2 for he2 in bdry if he1.vertices[0] == he2.vertices[1])
            he1.nxt, he1.prev = (nxt._heid, prev._heid)
        except StopIteration:
            print("Corner detected")
    # turn into dict for easy access
    return {he._heid: he for he in hes}

# %% ../00_triangle_data_structure.ipynb 51
class HalfEdgeMesh:
    def __init__(self, mesh : ListOfVerticesAndFaces):
        hes = get_half_edges(mesh)
        self.hes = hes
        self.faces = {key: Face(key, []) for key in mesh.faces.keys()}
        [self.faces[he.face].hes.append(he) for he in hes.values() if he.face is not None]        
        self.vertices = {key: Vertex(key, val, []) for key, val in mesh.vertices.items()}
        [self.vertices[he.vertices[1]].incident.append(he) for he in hes.values()]
        #self.edges = {min(he._heid, he.twin): Edge(he._heid, (he, hes[he.twin]), {"flipped": False})
        #              for he in hes.values() if he.vertices[0] < he.vertices[1]}
    
    def __deepcopy__(self):
        pass
    
    def to_ListOfVerticesAndFaces(self): # also not efficient
        points = {key: val.coords for key, val in self.vertices.items()}
        faces = {key: set(flatten([he.vertices for he in val.hes]))
                 for key, val in self.faces.items()}
        return ListOfVerticesAndFaces(points, faces)
    
    def saveObj(self, fname):
        self.to_ListOfVerticesAndFaces().saveObj(fname)
    
    @staticmethod
    def fromObj(fname):
        return HalfEdgeMesh(ListOfVerticesAndFaces.fromObj(fname))

# %% ../00_triangle_data_structure.ipynb 67
@patch
def reset_hes(self: HalfEdgeMesh, face_or_vertex: Union[Face, Vertex]):
    """Re-create the full list of half edges belonging to a face or vertex based on its first half edge.
    Note: for vertices, this relies on the mesh being a triangulation. If that's not the case, would
    need use different method (go around face, with special case for bdry)."""
    returned = False
    new_hes = []
    if isinstance(face_or_vertex, Face):
        start_he = face_or_vertex.hes[0]
        he = start_he
        while not returned:
            he = self.hes[he.nxt]
            new_hes.append(he)
            returned = (he == start_he)
        face_or_vertex.hes = new_hes
    if isinstance(face_or_vertex, Vertex):
        start_he = face_or_vertex.incident[0]
        he = start_he
        while not returned:
            he = self.hes[self.hes[he.nxt].twin]
            new_hes.append(he)
            returned = (he == start_he)
        face_or_vertex.incident = new_hes

# %% ../00_triangle_data_structure.ipynb 71
@patch
def flip_edge(self: HalfEdgeMesh, e: int):
    """Flip edge of a triangle mesh. Call by using he index
    If the two adjacent faces are not triangles, it does not work!
    For variable name convention, see jerryyin.info/geometry-processing-algorithms/half-edge/"""
    # collect the required objects
    if self.hes[e].face is None or self.hes[self.hes[e].twin].face is None:
        raise ValueError('Cannot flip boundary edge')
    # by convention, always flip the edge with min index
    e = min(e, self.hes[e].twin)
    e = self.hes[e]
    e5 = self.hes[e.prev]
    e4 = self.hes[e.nxt]
    twin = self.hes[e.twin]
    e1 = self.hes[twin.prev]
    e0 = self.hes[twin.nxt]
    # making sure the vertices and faces do not refer to any of the edges to be modified.
    f0, f1 = [self.faces[e1.face], self.faces[e5.face]]
    f0.hes, f1.hes = [[e1], [e5]]
    v3, v4, v2, v1 = [self.vertices[he.vertices[1]] for he in [e0, e1, e4, e5]]
    v3.incident, v4.incident, v2.incident, v1.incident = [[he] for he in [e0, e1, e4, e5]]
    # recycle e, twin.
    e.nxt = e5._heid
    e.prev = e0._heid
    e.face = f1._fid
    e.vertices = (v3._vid, v2._vid)
    twin.nxt = e1._heid
    twin.prev = e4._heid
    twin.face = f0._fid
    twin.vertices = (v2._vid, v3._vid)
    # update next/prev values for the external edges
    e0.nxt = e._heid
    e1.nxt = e4._heid
    e4.nxt = twin._heid
    e5.nxt = e0._heid
    e0.prev = e5._heid
    e1.prev = twin._heid
    e4.prev = e1._heid
    e5.prev = e._heid
    # update the face of e0, e4
    e0.face = f1._fid
    e4.face = f0._fid
    # re-complete the list of half-edges for the vertices and faces
    for vertex_or_face in [f0, f1]+[v3, v4, v2, v1]:
        self.reset_hes(vertex_or_face)
    # notifiy edge! return the _heid, or set "Flipped" or something!
    e.flipped, twin.flipped = (True, True)
        

# %% ../00_triangle_data_structure.ipynb 83
@patch
def is_consistent(self: HalfEdgeMesh):
    """For debugging/testing purposes"""
    # check next and prev relations are consistent with vertices
    assert all([he.vertices[1] == self.hes[he.nxt].vertices[0]
                and he.vertices[0] == self.hes[he.prev].vertices[1]
                for he in self.hes.values()])
    # check half edges are registered in cells
    assert all([he in self.faces[he.face].hes
                for he in self.hes.values() if he.face is not None])
    # check half edges are registered in vertices
    assert all([he in self.vertices[he.vertices[1]].incident
                for he in self.hes.values()])
    # check twins have matching vertices
    assert all([he.vertices == self.hes[he.twin].vertices[::-1]
                for he in self.hes.values()])
    # check everybody is a triangle
    assert all([len(fc.hes) == 3 for fc in self.faces.values()])
    
    return True

# %% ../00_triangle_data_structure.ipynb 87
@patch
def triplot(self: HalfEdgeMesh):
    """wraps plt.triplot"""
    list_format = self.to_ListOfVerticesAndFaces()
    fcs = np.array(list(list_format.faces.values()))
    pts = np.array(list(list_format.vertices.values())).T
    plt.triplot(pts[0], pts[1], fcs)

# %% ../00_triangle_data_structure.ipynb 92
@patch
def get_edge_vecs(self: HalfEdgeMesh):
    return {key: self.vertices[val.vertices[1]].coords-self.vertices[val.vertices[0]].coords
            for key, val in self.hes.items()}

@patch
def get_edge_lens(self: HalfEdgeMesh):
    return {key: np.linalg.norm(self.vertices[val.vertices[1]].coords-self.vertices[val.vertices[0]].coords)
            for key, val in self.hes.items()}

@patch
def set_rest_lengths(self: HalfEdgeMesh):
    for key, val in get_edge_lens.items():
        self.hes[key].rest =  val
