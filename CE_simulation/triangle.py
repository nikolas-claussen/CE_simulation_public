# AUTOGENERATED! DO NOT EDIT! File to edit: ../00_triangle_data_structure.ipynb.

# %% auto 0
__all__ = ['flatten', 'sort_vertices', 'sort_ids_by_vertices', 'get_neighbors', 'ListOfVerticesAndFaces', 'get_test_mesh',
           'HalfEdge', 'Vertex', 'Face', 'get_half_edges', 'HalfEdgeMesh', 'get_test_hemesh', 'get_test_mesh_large',
           'get_test_hemesh_large', 'get_boundary_faces', 'load_mesh']

# %% ../00_triangle_data_structure.ipynb 3
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

from scipy import spatial

# %% ../00_triangle_data_structure.ipynb 4
from collections import defaultdict

# %% ../00_triangle_data_structure.ipynb 5
from dataclasses import dataclass
from typing import Union, Dict, List, Tuple, Iterable, Callable
from nptyping import NDArray, Int, Float, Shape

from fastcore.foundation import patch

# %% ../00_triangle_data_structure.ipynb 7
from bisect import bisect_left

# %% ../00_triangle_data_structure.ipynb 9
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

# %% ../00_triangle_data_structure.ipynb 10
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

# %% ../00_triangle_data_structure.ipynb 18
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

# %% ../00_triangle_data_structure.ipynb 19
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
    
    def remove_face(self, f_id):
        face = self.vertices[f_id]
        del self.faces[f_id]
        for v in face:
            if not any([v in fc for fc in self.faces.values()]):
                del self.vertices[v]
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

# %% ../00_triangle_data_structure.ipynb 20
def get_test_mesh():
    points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
    tri = spatial.Delaunay(points)
    return ListOfVerticesAndFaces(tri.points, tri.simplices)

# %% ../00_triangle_data_structure.ipynb 28
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


# %% ../00_triangle_data_structure.ipynb 35
from dataclasses import dataclass, field

# %% ../00_triangle_data_structure.ipynb 36
@dataclass
class HalfEdge:
    """Attribute holder class for half edges. Attributes point to other items, property methods get them."""
    _heid : int
    _nxtid: int
    _previd: int
    _twinid: int
    _faceid: Union[int, None] # None if it's a boundary
    _verticesid: tuple # 0 is origin, 1 is destination
    rest: float = 0.
    passive: float = 0.
    variables: dict = field(default_factory=dict, repr=False) 
    _hemesh: int = field(default=None, repr=False) # set during creation of mesh
    duplicate: bool = False # arbitraily select half of all edges for future iteration convenience
    # set methods to get the twins, nxts, and prevs using the internal use _ids to look them up in the dict
    @property
    def nxt(self):
        return self._hemesh.hes[self._nxtid]
    @property
    def prev(self):
        return self._hemesh.hes[self._previd]
    @property
    def twin(self):
        return self._hemesh.hes[self._twinid]
    @property
    def face(self):
        if self._faceid is None:
            return None
        return self._hemesh.faces[self._faceid]
    @property
    def vertices(self):
        return [self._hemesh.vertices[v] for v in self._verticesid]
    def __post_init__(self):
        self.duplicate = self._verticesid[0] < self._verticesid[1]
        
    def __repr__(self):
        repr_str = f"HalfEdge(heid={self._heid}, nxt={self._nxtid}, prev={self._previd}, twin={self._twinid}, "
        if self._faceid is not None:
            repr_str += f"face={self._faceid}, "
        else:
            repr_str += f"face=None, "
        repr_str += f"vertices={self._verticesid}, "
        repr_str += f"rest={round(self.rest, ndigits=1)}, passive={round(self.passive, ndigits=1)}"
        if self._hemesh is not None:
            repr_str += f", center={np.round(np.mean([v.coords  for v in self.vertices], axis=0), decimals=1)}"
        return repr_str

    
@dataclass
class Vertex:
    """Attribute holder class for vertices. Attributes point to other items. Note: different from the
    standard half edge data structure, I store all incident he's, for latter convenience (e.g. force balance)
    computation."""
    _vid : int
    coords : NDArray[Shape["2"], Float]
    incident : List[HalfEdge]
    rest_shape: NDArray[Shape["2, 2"],Float] = np.array([[1.0, 0.0], [0.0, 1.0]])
    def __repr__(self):
        repr_str = f"Vertex(vid={self._vid}, coords={np.round(self.coords, decimals=1)}, "
        repr_str += f"hes={[he._heid for he in self.incident]})"
        return repr_str
        

@dataclass
class Face:
    """Attribute holder class for faces. Attributes point to other items."""
    _fid : int
    hes : List[HalfEdge]
    dual_coords: Union[NDArray[Shape["2"],Float], None] = None
    rest_shape: NDArray[Shape["2, 2"],Float] = np.array([[1.0, 0.0], [0.0, 1.0]])
    def __repr__(self):
        repr_str = f"Face(fid={self._fid}, "
        if self.dual_coords is not None:
            repr_str += f"dual_coords={list(np.round(self.dual_coords, decimals=1))}, "
        repr_str += f"rest_shape={[list(x) for x in np.round(self.rest_shape, decimals=1)]}, "
        repr_str += f"hes={[he._heid for he in self.hes]})"
        return repr_str


# %% ../00_triangle_data_structure.ipynb 39
@patch
def sort_hes(self: Face):
    """Sort the list of hes of a face."""
    sorted_hes = []
    returned = False
    start_he = self.hes[0]
    he = start_he
    while not returned:
        sorted_hes.append(he)
        if he._hemesh is not None:
            he = he.nxt
        else:
            he = next(x for x in self.hes if x._heid == he._nxtid)
        returned = (he == start_he)
    self.hes = sorted_hes

# %% ../00_triangle_data_structure.ipynb 44
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
        nxts, prevs = (np.roll(heids, -1).tolist(), np.roll(heids, +1).tolist())
        vertices = [tuple((fc+[fc[0]])[i:i+2]) for i in range(len(fc))]
        for _heid, _nxtid, _previd, _verticesid in zip(heids, nxts, prevs, vertices):
             he_vertex_dict[_verticesid] = HalfEdge(_heid, _nxtid, _previd, None, key, _verticesid)
                # is the order correct here??
        heid_counter += len(fc)
    # now match the half-edges. if they cannot match, add a new he with faec None
    hes = []
    for he1 in he_vertex_dict.values():
        try:
            he2 = he_vertex_dict[he1._verticesid[::-1]]
        except KeyError:
            he2 = HalfEdge(heid_counter, None, None, he1._heid, None, he1._verticesid[::-1],)
            heid_counter += 1
        he1._twinid, he2._twinid = (he2._heid, he1._heid)
        hes.append(he1); hes.append(he2)
    # find the "next" of the boundary edges. we can just traverse inshallah
    bdry = [he for he in hes if he._faceid is None]
    for he1 in bdry:
        try:
            nxt = next(he2 for he2 in bdry if he1._verticesid[1] == he2._verticesid[0])
            prev = next(he2 for he2 in bdry if he1._verticesid[0] == he2._verticesid[1])
            he1._nxtid, he1._previd = (nxt._heid, prev._heid)
        except StopIteration:
            print("Corner detected")
    # turn into dict for easy access
    return {he._heid: he for he in hes}

# %% ../00_triangle_data_structure.ipynb 47
class HalfEdgeMesh:
    def __init__(self, mesh : ListOfVerticesAndFaces):
        hes = get_half_edges(mesh)
        self.hes = hes
        self.faces = {key: Face(key, []) for key in mesh.faces.keys()}
        [self.faces[he._faceid].hes.append(he) for he in hes.values() if he._faceid is not None]
        self.vertices = {key: Vertex(key, val, []) for key, val in mesh.vertices.items()}
        [self.vertices[he._verticesid[1]].incident.append(he) for he in hes.values()]
        for he in self.hes.values():
            he._hemesh = self
        [fc.sort_hes() for fc in self.faces.values()]

    
    #def __deepcopy__(self):
    #    pass
    
    def to_ListOfVerticesAndFaces(self): # also not efficient
        vertices = {key: val.coords for key, val in self.vertices.items()}
        faces = {key: set(flatten([he._verticesid for he in val.hes]))
                 for key, val in self.faces.items()}
        return ListOfVerticesAndFaces(vertices, faces)
    
    def saveObj(self, fname):
        self.to_ListOfVerticesAndFaces().saveObj(fname)
    
    @staticmethod
    def fromObj(fname):
        return HalfEdgeMesh(ListOfVerticesAndFaces.fromObj(fname))

# %% ../00_triangle_data_structure.ipynb 48
def get_test_hemesh():
    return HalfEdgeMesh(get_test_mesh())

# %% ../00_triangle_data_structure.ipynb 61
def get_test_mesh_large(x=np.linspace(0, 1, 25), y=np.linspace(0, 1, 50), noise=.0025):
    pts = np.stack(np.meshgrid(x, y))
    
    np.random.seed(1241) # get consistent results
    noise =  np.random.normal(size=pts.shape, scale=noise)
    noise[:,0,:] = noise[:,-1,:] = 0
    noise[:,:,0] = noise[:,:,-1] = 0
    pts += noise
    
    pts = pts.reshape((2, pts.shape[1]*pts.shape[2])).T
    tri = spatial.Delaunay(pts)
    return ListOfVerticesAndFaces(tri.points, tri.simplices)

def get_test_hemesh_large(x=np.linspace(0, 1, 25), y=np.linspace(0, 1, 50), noise=.0025):
    return HalfEdgeMesh(get_test_mesh_large(x=x, y=y, noise=noise))

# %% ../00_triangle_data_structure.ipynb 68
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
            he = he.nxt
            new_hes.append(he)
            returned = (he == start_he)
        face_or_vertex.hes = new_hes
    if isinstance(face_or_vertex, Vertex):
        start_he = face_or_vertex.incident[0]
        he = start_he
        while not returned:
            he = he.nxt.twin
            new_hes.append(he)
            returned = (he == start_he)
        face_or_vertex.incident = new_hes

# %% ../00_triangle_data_structure.ipynb 72
@patch
def flip_edge(self: HalfEdgeMesh, e: int):
    """Flip edge of a triangle mesh. Call by using he index
    If the two adjacent faces are not triangles, it does not work!
    For variable name convention, see jerryyin.info/geometry-processing-algorithms/half-edge/"""
    # by convention, always flip the duplicate
    e = self.hes[e]
    e = e if e.duplicate else e.twin
    if e._faceid is None or e.twin._faceid is None:
        raise ValueError('Cannot flip boundary edge')
    # collect the required objects
    e5 = e.prev
    e4 = e.nxt
    twin = e.twin
    e1 = twin.prev
    e0 = twin.nxt
    # making sure the vertices and faces do not refer to any of the edges to be modified.
    f0, f1 = [e1.face, e5.face]
    f0.hes, f1.hes = [[e1], [e5]]
    v3, v4, v2, v1 = [he.vertices[1] for he in [e0, e1, e4, e5]]
    v3.incident, v4.incident, v2.incident, v1.incident = [[he] for he in [e0, e1, e4, e5]]
    # recycle e, twin.
    e._nxtid = e5._heid
    e._previd = e0._heid
    e._faceid = f1._fid
    e._verticesid = (v3._vid, v2._vid)
    twin._nxtid = e1._heid
    twin._previd = e4._heid
    twin._faceid = f0._fid
    twin._verticesid = (v2._vid, v3._vid)
    # update next/prev values for the external edges
    e0._nxtid = e._heid
    e1._nxtid = e4._heid
    e4._nxtid = twin._heid
    e5._nxtid = e0._heid
    e0._previd = e5._heid
    e1._previd = twin._heid
    e4._previd = e1._heid
    e5._previd = e._heid
    # update the face of e0, e4
    e0._faceid = f1._fid
    e4._faceid = f0._fid
    # re-complete the list of half-edges for the vertices and faces
    for vertex_or_face in [f0, f1]+[v3, v4, v2, v1]:
        self.reset_hes(vertex_or_face)
    # re-order the faces
    f0.sort_hes(); f1.sort_hes()
    
        

# %% ../00_triangle_data_structure.ipynb 84
@patch
def is_consistent(self: HalfEdgeMesh):
    """For debugging/testing purposes"""
    # check next and prev relations are consistent with vertices
    assert all([he.vertices[1] == he.nxt.vertices[0]
                and he.vertices[0] == he.prev.vertices[1]
                for he in self.hes.values()])
    # check half edges are registered in cells
    assert all([he in he.face.hes
                for he in self.hes.values() if he.face is not None])
    # check half edges are registered in vertices
    assert all([he in he.vertices[1].incident
                for he in self.hes.values()])
    # check twins have matching vertices
    assert all([he.vertices == he.twin.vertices[::-1]
                for he in self.hes.values()])
    # check everybody is a triangle
    assert all([len(fc.hes) == 3 for fc in self.faces.values()])
    # check all triangles are sorted
    assert all([all([(fc.hes[i].nxt == fc.hes[(i+1)%3]) for i in range(3)])
                for fc in self.faces.values()])
    
    return True

# %% ../00_triangle_data_structure.ipynb 90
@patch
def is_bdr(self: Face):
    """True if face touches bdr. Check all vertices. Does any have an incident edge with None face?"""
    verts = [he.vertices[1] for he in self.hes]
    return any([any([he.face is None for he in v.incident]) for v in verts])

def get_boundary_faces(msh):
    """Get indices of boundary faces"""
    bdr_faces = []
    bdr_start = next(he for he in msh.hes.values() if he.face is None).twin.nxt
    he = bdr_start
    returned = False
    while not returned:
        bdr_faces.append(he.face._fid)
        if he.nxt.twin.face.is_bdr():
            he = he.nxt.twin
        else:
            he = he.prev.twin
        returned = (he == bdr_start)
    return bdr_faces

# %% ../00_triangle_data_structure.ipynb 91
@patch
def get_face_neighbors(self: Vertex):
    """Get face neighbors of vertex"""
    neighbors = []
    start_he = self.incident[0]
    he = start_he
    returned = False
    while not returned:
        neighbors.append(he.face)
        he = he.nxt.twin
        returned = (he == start_he)
    return neighbors


# %% ../00_triangle_data_structure.ipynb 93
@patch
def set_centroid(self: HalfEdgeMesh):
    """Set dual positions to triangle centroid"""
    for fc in self.faces.values():
        vecs = []
        returned = False
        start_he = fc.hes[0]
        he = start_he
        while not returned:
            vecs.append(he.vertices[0].coords)
            he = he.nxt
            returned = (he == start_he)
        fc.dual_coords = np.mean(vecs, axis=0)

# %% ../00_triangle_data_structure.ipynb 95
@patch
def transform_vertices(self: HalfEdgeMesh, trafo: Union[Callable, NDArray[Shape["2, 2"], Float]]):
    for v in self.vertices.values():
        if isinstance(trafo, Callable):
            v.coords = trafo(v.coords)
        else:
            v.coords = trafo.dot(v.coords)
            
@patch
def transform_dual_vertices(self: HalfEdgeMesh, trafo: Union[Callable, NDArray[Shape["2, 2"], Float]]):
    for fc in self.faces.values():
        if isinstance(trafo, Callable):
            fc.dual_coords = trafo(fc.dual_coords)
        else:
            fc.dual_coords = trafo.dot(fc.dual_coords)

# %% ../00_triangle_data_structure.ipynb 97
@patch
def triplot(self: HalfEdgeMesh):
    """wraps plt.triplot"""
    list_format = self.to_ListOfVerticesAndFaces()
    fcs = np.array(list(list_format.faces.values()))
    pts = np.array(list(list_format.vertices.values())).T
    plt.triplot(pts[0], pts[1], fcs)
    
@patch
def labelplot(self: HalfEdgeMesh, vertex_labels=True, face_labels=True,
                     halfedge_labels=False, cell_labels=False):
    """for debugging purposes, a fct to plot a trimesh with labels attached"""
    if face_labels:
        for fc in self.faces.values():
            centroid = np.mean([he.vertices[0].coords for he in fc.hes], axis=0)
            plt.text(*centroid, str(fc._fid), color="k")
    if vertex_labels:
        for v in self.vertices.values():
            plt.text(*(v.coords+np.array([0,.05])), str(v._vid),
                     color="tab:blue", ha="center")
    if cell_labels:
        for v in self.vertices.values():
            nghbs = v.get_face_neighbors()
            if not (None in nghbs):
                center = np.mean([fc.dual_coords for fc in nghbs], axis=0)
                plt.text(*(center), str(v._vid),
                         color="tab:blue", ha="center")
    if halfedge_labels:
        for he in self.hes.values():
            if he.duplicate:
                centroid = np.mean([v.coords for v in he.vertices], axis=0)
                plt.text(*centroid, str(he._heid), color="tab:orange")
                
@patch
def cellplot(self: HalfEdgeMesh, alpha=1, set_lims=False, edge_colors=None, cell_colors=None):
    """Plot based on primal positions. Now fast because of use of LineCollection.
    edge_colors, cell_colors are dicts _heid, _vid : color. Need only specify non-default elements.
    """
    face_keys = sorted(self.faces.keys())
    face_key_dict = {key: ix for ix, key in enumerate(face_keys)}
    face_key_dict[None] = None
    primal_face_list = []
    
    reformated_cell = (defaultdict(lambda: (0,0,0,0)) if cell_colors is None
                       else defaultdict(lambda: (0,0,0,0), cell_colors))
    facecolors = []
    for v in self.vertices.values():
        neighbors = v.get_face_neighbors()
        if not (None in neighbors):
            facecolors.append(reformated_cell[v._vid])
            face = [face_key_dict[fc._fid] for fc in neighbors]
            face.append(face[0])
            primal_face_list.append(face)
    
    pts = np.stack([self.faces[key].dual_coords for key in face_keys])
    lines = flatten([[[pts[a],pts[b]] for a, b  in zip(fc, np.roll(fc, 1))]
                     for fc in primal_face_list], max_depth=1)
    cells = [[pts[v] for v in fc] for fc in primal_face_list]
    
    reformated_edge = defaultdict(lambda: "k")
    if edge_colors is not None: # translate from _heid : color to 
        for key, val in edge_colors.items():
            he = self.hes[key]
            if (he.face is not None) and (he.twin.face is not None):
                newkey = (face_key_dict[he.face._fid], face_key_dict[he.twin.face._fid])
                reformated_edge[newkey] = reformated_edge[newkey[::-1]] = val
    colors = list(flatten([[reformated_edge[(a, b)] for a, b  in zip(fc, np.roll(fc, 1))]
                            for fc in primal_face_list], max_depth=1))
    
    #fig, ax = plt.subplots()
    plt.gca().add_collection(LineCollection(lines, colors=colors, alpha=alpha))
    if cell_colors is not None:
        plt.gca().add_collection(LineCollection(cells, facecolors=facecolors,
                                                colors=(0,0,0,0)))
    
    if set_lims:
        plt.gca().set_xlim([pts[:,0].min(), pts[:,0].max()])
        plt.gca().set_ylim([pts[:,1].min(), pts[:,1].max()])

# %% ../00_triangle_data_structure.ipynb 103
@patch
def save_mesh(self: HalfEdgeMesh, fname, d=5):
    """Save HalfEdgeMesh in as csv file with 3 parts:
    1. Dual vertices
        - vertex id (int)
        - vertex coordinates x, y
        - one incident edge id
    2. Faces (triangles)
        - face id
        - vertex ids 1-3
        - dual coords x,y
        - one edge id
    3. Half-edges
        - half-edge ID
        - vertex ids 1-2
        - face ids 1-2 (i.e. its own face + its twin)
        - next, prev, twin
    See jerryyin.info/geometry-processing-algorithms/half-edge/ for definitions.
    '#' are comment lines.
    """
    # overwrite
    try:
        os.remove(fname)
    except OSError:
        pass
    v_keys = sorted(self.vertices.keys())
    fc_keys = sorted(self.faces.keys())
    he_keys = sorted(self.hes.keys())

    with open(fname, "a") as f:
        f.write('# Vertices\n')
        f.write('# vertex id, dual x-coordinate, dual y-coordinate, incident edge id\n')
        for key in v_keys:
            v = self.vertices[key]
            items = ([v._vid]
                     +[round(v.coords[0], ndigits=d), round(v.coords[1], ndigits=d)]
                     +[v.incident[0]._heid])
            to_write = ', '.join([str(x) for x in items]) + '\n'
            f.write(to_write)
        f.write('\n# Faces\n')
        f.write('# face id, primal x-coordinate, primal y-coordinate, vertex id 1, vertex id 2, vertex id 3, edge 1, edge 2, edge 3\n')
        for key in fc_keys:
            fc = self.faces[key]
            items = ([fc._fid]
                     +[round(fc.dual_coords[0], ndigits=d), round(fc.dual_coords[1], ndigits=d)]
                     +[he.vertices[0]._vid for he in fc.hes]
                     +[he._heid for he in fc.hes])
            to_write = ', '.join([str(x) for x in items]) + '\n'
            f.write(to_write)
        f.write('\n# Half-edges\n')
        f.write('# edge id, vertex id 1, vertex id 2, face id 1, face id 2, next, prev, twin\n')
        for key in he_keys:
            he = self.hes[key]
            items = ([he._heid]
                     +[v._vid for v in he.vertices]
                     +[(fc._fid if fc is not None else "None") for fc in [he.face, he.twin.face]]
                     +[x._heid for x in [he.nxt, he.prev, he.twin]])
            to_write = ', '.join([str(x) for x in items]) + '\n'
            f.write(to_write)
            
            

# %% ../00_triangle_data_structure.ipynb 105
def load_mesh(fname):
    """Load from file as saved by mesh.save_mesh"""
    with open(fname) as f:
        lines = f.readlines()
    vind, find, heind = [lines.index(x) for x in ['# Vertices\n', '# Faces\n', '# Half-edges\n',]]
    vlines, flines, helines = [lines[vind+2:find-1], lines[find+2:heind-1], lines[heind+2:]]
    vlines, flines, helines = [[x.strip('\n').split(', ') for x in y] for y in [vlines, flines, helines]]
    vlines = [[int(x[0]), float(x[1]), float(x[2]), int(x[3])] for x in vlines]
    flines = [[int(x[0]), float(x[1]), float(x[2]), int(x[3]), int(x[4]), int(x[5]),
               int(x[6]), int(x[7]), int(x[8])] for x in flines]
    int_None = lambda x: int(x) if x != 'None' else None
    helines = [[int(x[0]), int(x[1]), int(x[2]), int_None(x[3]), int_None(x[4]),
                int(x[5]), int(x[6]), int(x[7])] for x in helines]

    hes = {x[0]: HalfEdge(_heid=x[0], _nxtid=x[5], _previd=x[6], _twinid=x[7], _faceid=x[3],
                          _verticesid=(x[1], x[2])) for x in helines}
    vertices = {x[0]: Vertex(_vid=x[0], coords=np.array([x[1], x[2]]), incident=[hes[x[-1]]]) for x in vlines}
    faces = {x[0]: Face(_fid=x[0], dual_coords=np.array([x[1], x[2]]), hes=[hes[x[-1]]] ) for x in flines}

    hemesh = HalfEdgeMesh(ListOfVerticesAndFaces([],[]))
    hemesh.hes = hes
    hemesh.vertices = vertices
    hemesh.faces = faces
    for he in hemesh.hes.values():
        he._hemesh = hemesh
    [hemesh.reset_hes(v) for v in hemesh.vertices.values()]
    [hemesh.reset_hes(fc) for fc in hemesh.faces.values()]
    return hemesh

# %% ../00_triangle_data_structure.ipynb 110
@patch
def get_edge_lens(self: HalfEdgeMesh):
    return {key: np.linalg.norm(val.vertices[1].coords-val.vertices[0].coords)
            for key, val in self.hes.items()}

@patch
def set_rest_lengths(self: HalfEdgeMesh):
    for key, val in self.get_edge_lens().items():
        self.hes[key].rest = val
        
@patch
def get_rel_tension(self: HalfEdgeMesh):
    rel_tensions = {}
    for he in self.hes.values():
        surrounding = []
        if he.duplicate and he.face is not None:
            surrounding.append(he.nxt.rest)
            surrounding.append(he.prev.rest)
            twin = he.twin
            surrounding.append(twin.nxt.rest)
            surrounding.append(twin.prev.rest)
            rel_tensions[he._heid], rel_tensions[twin._heid] = 2*(4*he.rest/sum(surrounding),)
    return rel_tensions
