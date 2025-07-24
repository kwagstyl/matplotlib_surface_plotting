"""Geometric calculations for surface plotting."""

import numpy as np
from typing import Tuple, List, Optional, Union


def normalize_v3(arr: np.ndarray) -> np.ndarray:
    """Normalize a numpy array of 3 component vectors shape=(n,3)."""
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def normal_vectors(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute face normal vectors for triangular mesh.
    
    Args:
        vertices: Array of vertex positions (n, 3)
        faces: Array of face indices (m, 3)
        
    Returns:
        Array of face normal vectors (m, 3)
    """
    tris = vertices[faces]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    return normalize_v3(n)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute vertex normal vectors by averaging adjacent face normals.
    
    Args:
        vertices: Array of vertex positions (n, 3)
        faces: Array of face indices (m, 3)
        
    Returns:
        Array of vertex normal vectors (n, 3)
    """
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    tris = vertices[faces]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n = normalize_v3(n)
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    return normalize_v3(norm)


def shading_intensity(vertices: np.ndarray, faces: np.ndarray, 
                     light: np.ndarray = np.array([0, 0, 1]), 
                     shading: float = 0.7) -> np.ndarray:
    """Calculate shading intensity based on light source.
    
    Args:
        vertices: Array of vertex positions (n, 3)
        faces: Array of face indices (m, 3)
        light: Light direction vector (3,)
        shading: Amount of shading to apply (0-1)
        
    Returns:
        Array of intensity values for each face
    """
    face_normals = normal_vectors(vertices, faces)
    intensity = np.dot(face_normals, light)
    intensity[np.isnan(intensity)] = 1
    
    # Top 20% all become fully coloured
    intensity = ((1 - shading) + shading * 
                 (intensity - np.min(intensity)) / 
                 (np.percentile(intensity, 80) - np.min(intensity)))
    
    # Saturate
    intensity[intensity > 1] = 1
    return intensity


def get_neighbours_from_tris(tris: np.ndarray, 
                           label: Optional[np.ndarray] = None) -> np.ndarray:
    """Get surface neighbours from triangulation.
    
    Args:
        tris: Triangle indices array (m, 3)
        label: Optional vertex labels to filter neighbours
        
    Returns:
        Array of neighbour lists for each vertex
    """
    n_vert = np.max(tris + 1)
    neighbours = [[] for i in range(n_vert)]
    
    for tri in tris:
        neighbours[tri[0]].extend([tri[1], tri[2]])
        neighbours[tri[2]].extend([tri[0], tri[1]])
        neighbours[tri[1]].extend([tri[2], tri[0]])
    
    # Get unique neighbours
    for k in range(len(neighbours)):
        if label is not None:
            neighbours[k] = set(neighbours[k]).intersection(label)
        else:
            neighbours[k] = _unique_ordered(neighbours[k])
    
    return np.array(neighbours, dtype=object)


def get_ring_of_neighbours(island: np.ndarray, neighbours: np.ndarray,
                          vertex_indices: Optional[np.ndarray] = None,
                          ordered: bool = False) -> np.ndarray:
    """Calculate ring of neighbouring vertices for an island of cortex.
    
    Args:
        island: Boolean array marking island vertices
        neighbours: Neighbour lists for each vertex
        vertex_indices: Optional vertex indices
        ordered: Whether to return vertices in connected order
        
    Returns:
        Array of neighbouring vertex indices
    """
    if vertex_indices is None:
        vertex_indices = np.arange(len(island))
    
    if not ordered:
        neighbours_island = neighbours[island]
        unfiltered_neighbours = []
        for n in neighbours_island:
            unfiltered_neighbours.extend(n)
        unique_neighbours = np.setdiff1d(
            np.unique(unfiltered_neighbours), 
            vertex_indices[island]
        )
        return unique_neighbours


def frontback(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort front and back facing triangles.
    
    Args:
        T: Triangles to sort (n, 3, 2)
        
    Returns:
        Tuple of (front_mask, back_mask) boolean arrays
    """
    Z = ((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 1, 1] + T[:, 0, 1]) +
         (T[:, 2, 0] - T[:, 1, 0]) * (T[:, 2, 1] + T[:, 1, 1]) +
         (T[:, 0, 0] - T[:, 2, 0]) * (T[:, 0, 1] + T[:, 2, 1]))
    return Z < 0, Z >= 0


def normalized(a: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:
    """Normalize array along specified axis."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def ras2coords(mri_img, ras: Tuple[float, float, float] = (0, 0, 0)) -> Tuple:
    """Convert RAS coordinates to image coordinates."""
    return tuple(mri_img.affine[:3, :3].dot(ras) + mri_img.affine[:3, 3])


def compute_plane_from_mri(mri_img, slice_i: int, slice_axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute plane coordinates from MRI slice.
    
    Args:
        mri_img: MRI image object with affine transformation
        slice_i: Slice index
        slice_axis: Axis along which to slice (0, 1, or 2)
        
    Returns:
        Tuple of (plane_coords, plane_faces)
    """
    shape = mri_img.shape
    
    if slice_axis == 0:
        coord = slice_i
        v0 = ras2coords(mri_img, (coord, 0, 0))
        v1 = ras2coords(mri_img, (coord, shape[1], 0))
        v2 = ras2coords(mri_img, (coord, 0, shape[2]))
        v3 = ras2coords(mri_img, (coord, shape[1], shape[2]))
    elif slice_axis == 1:
        coord = slice_i
        v0 = ras2coords(mri_img, (0, coord, 0))
        v1 = ras2coords(mri_img, (shape[0], coord, 0))
        v2 = ras2coords(mri_img, (0, coord, shape[2]))
        v3 = ras2coords(mri_img, (shape[0], coord, shape[2]))
    else:
        coord = slice_i
        v0 = ras2coords(mri_img, (0, 0, coord))
        v1 = ras2coords(mri_img, (shape[0], 0, coord))
        v2 = ras2coords(mri_img, (0, shape[1], coord))
        v3 = ras2coords(mri_img, (shape[0], shape[1], coord))
    
    return np.array([v0, v1, v2, v3]), np.array([[0, 1, 2], [1, 3, 2]])


def bounds_from_mesh(axis: int, coord: float, mesh_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create plane bounds from mesh dimensions.
    
    Args:
        axis: Plane axis (0, 1, or 2)
        coord: Coordinate value along axis
        mesh_vertices: Mesh vertex array
        
    Returns:
        Tuple of (plane_coords, plane_faces)
    """
    min_vals = mesh_vertices.min(axis=0)
    max_vals = mesh_vertices.max(axis=0)
    # Border is 10% of the mesh size
    border = 0.1 * (max_vals - min_vals)
    min_vals -= border
    max_vals += border
    
    if axis == 0:
        coords = np.array([
            [coord, min_vals[1], min_vals[2]],
            [coord, max_vals[1], min_vals[2]],
            [coord, min_vals[1], max_vals[2]],
            [coord, max_vals[1], max_vals[2]]
        ])
    elif axis == 1:
        coords = np.array([
            [min_vals[0], coord, min_vals[2]],
            [max_vals[0], coord, min_vals[2]],
            [min_vals[0], coord, max_vals[2]],
            [max_vals[0], coord, max_vals[2]]
        ])
    else:  # axis == 2
        coords = np.array([
            [min_vals[0], min_vals[1], coord],
            [max_vals[0], min_vals[1], coord],
            [min_vals[0], max_vals[1], coord],
            [max_vals[0], max_vals[1], coord]
        ])
    
    return coords, np.array([[0, 1, 2], [1, 3, 2]])


def get_plane_intersections(mesh_vertices: np.ndarray, mesh_faces: np.ndarray,
                          plane_coords: np.ndarray, plane_faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Categorize mesh faces relative to a plane.
    
    Args:
        mesh_vertices: Mesh vertex array (N, 3)
        mesh_faces: Mesh face indices (M, 3)
        plane_coords: Plane vertex array (P, 3)
        plane_faces: Plane face indices (Q, 3)
        
    Returns:
        Tuple of (behind_faces, intersected_faces, in_front_faces) boolean masks
    """
    # Compute plane normal using the first triangle of the plane
    idx0, idx1, idx2 = plane_faces[0]
    v0 = plane_coords[idx0]
    v1 = plane_coords[idx1]
    v2 = plane_coords[idx2]
    
    # Edge vectors
    e1 = v1 - v0
    e2 = v2 - v0
    plane_normal = np.cross(e1, e2)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    plane_point = v0  # any point on the plane

    behind_faces = np.zeros(len(mesh_faces), dtype=bool)
    intersected_faces = np.zeros(len(mesh_faces), dtype=bool)
    in_front_faces = np.zeros(len(mesh_faces), dtype=bool)

    for i, face in enumerate(mesh_faces):
        verts = mesh_vertices[face]
        signed_dists = np.dot(verts - plane_point, plane_normal)

        if np.all(signed_dists < 0):
            behind_faces[i] = True
        elif np.all(signed_dists > 0):
            in_front_faces[i] = True
        else:
            intersected_faces[i] = True

    return behind_faces, intersected_faces, in_front_faces


def _unique_ordered(seq: List) -> List:
    """Return unique elements preserving order."""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]