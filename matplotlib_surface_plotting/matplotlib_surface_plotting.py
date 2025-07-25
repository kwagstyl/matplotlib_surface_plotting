"""Matplotlib-based surface plotting for neuroimaging data."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from typing import List, Optional, Dict, Tuple

from .transformations import create_mvp_matrix, yrotate
from .geometry import (
    shading_intensity, vertex_normals, frontback, 
    compute_plane_from_mri, bounds_from_mesh, get_plane_intersections
)
from .colors import (
    mask_colours, adjust_colours_pvals, add_parcellation_colours,
    adjust_colours_alpha, apply_intensity_shading, create_colormap_colours,
    set_random_seed
)


def _prepare_surface_data(vertices: np.ndarray, faces: np.ndarray, 
                         overlays: List[np.ndarray], plane_config: Optional[Dict]) -> Tuple:
    """Prepare and normalize surface data for rendering."""
    vertices = vertices.astype(np.float32)
    faces = faces.astype(int)
    
    # Handle plane normalization
    if plane_config is not None:
        if plane_config['mri_img'] is None:
            plane_coords, plane_faces = bounds_from_mesh(
                plane_config['slice_axis'], plane_config['slice_i'], vertices)
        else:
            plane_coords, plane_faces = compute_plane_from_mri(
                plane_config['mri_img'], plane_config['slice_i'], plane_config['slice_axis'])
        
        # Normalize both mesh and plane to same coordinate system
        center = (plane_coords.max(0) + plane_coords.min(0)) / 2
        scale = max(plane_coords.max(0) - plane_coords.min(0))
        vertices = (vertices - center) / scale
        plane_coords = (plane_coords - center) / scale
        
        return vertices, faces, overlays, (plane_coords, plane_faces)
    else:
        # Standard mesh normalization
        center = (vertices.max(0) + vertices.min(0)) / 2
        scale = max(vertices.max(0) - vertices.min(0))
        vertices = (vertices - center) / scale
        
        return vertices, faces, overlays, None


def _setup_lighting_and_plane(vertices: np.ndarray, faces: np.ndarray, 
                              plane_data: Optional[Tuple], z_rotate: float,
                              flat_map: bool, plane_config: Optional[Dict]) -> Tuple:
    """Setup lighting and plane intersection data."""
    if flat_map:
        intensity = np.ones(len(faces))
        plane_intersections = None
        plane_colours = None
    else:
        # Calculate lighting
        light = np.array([0, 0, 1, 1]) @ yrotate(z_rotate)
        intensity = shading_intensity(vertices, faces, light=light[:3], shading=0.7)
        
        # Handle plane intersections
        if plane_data is not None:
            plane_coords, plane_faces = plane_data
            behind_faces, intersected_faces, in_front_faces = get_plane_intersections(
                vertices, faces, plane_coords, plane_faces)
            
            # Calculate plane intensity (for future use)
            _ = shading_intensity(plane_coords, plane_faces, 
                                light=light[:3], shading=0.7)
            plane_colours = np.ones((len(plane_faces), 4))
            plane_colours[:, :3] = plane_config['plane_colour'][:3]
            plane_colours[:, 3] = plane_config['plane_alpha']
            
            plane_intersections = {
                'behind': behind_faces,
                'intersected': intersected_faces, 
                'in_front': in_front_faces,
                'coords': plane_coords,
                'faces': plane_faces
            }
        else:
            plane_intersections = None
            plane_colours = None
    
    return intensity, plane_intersections, plane_colours


def _process_overlay_colours(overlay: np.ndarray, faces: np.ndarray, cmap: str,
                           vmin: Optional[float], vmax: Optional[float], label: bool,
                           alpha_colour: Optional[np.ndarray], pvals: Optional[np.ndarray],
                           mask: Optional[np.ndarray], mask_colour: Optional[np.ndarray],
                           border_colour: np.ndarray, parcel: Optional[np.ndarray],
                           parcel_cmap: Optional[Dict], filled_parcels: bool,
                           neighbours: Optional[np.ndarray],
                           plane_intersections: Optional[Dict],
                           plane_config: Optional[Dict]) -> np.ndarray:
    """Process overlay data into face colours."""
    # Create base colours from overlay
    colours, vmin_used, vmax_used = create_colormap_colours(
        overlay, faces, cmap, vmin, vmax, label)
    
    # Apply alpha transparency
    if alpha_colour is not None:
        face_alpha = np.mean(alpha_colour[faces], axis=1) if not label else np.median(alpha_colour[faces], axis=1)
        colours = adjust_colours_alpha(colours, face_alpha)
    
    # Apply p-value adjustments
    if pvals is not None:
        colours = adjust_colours_pvals(colours, pvals, faces, mask, 
                                     mask_colour=mask_colour, 
                                     border_colour=border_colour)
    elif mask is not None:
        colours = mask_colours(colours, faces, mask, mask_colour=mask_colour)
    
    # Add parcellation
    if parcel is not None:
        colours = add_parcellation_colours(colours, parcel, faces, parcel_cmap,
                                         mask, mask_colour=mask_colour,
                                         filled=filled_parcels, 
                                         neighbours=neighbours)
    
    # Mark plane intersections
    if plane_intersections is not None and plane_config is not None:
        colours[plane_intersections['intersected'], :] = plane_config['intersected_colour']
    
    return colours, vmin_used, vmax_used


def _transform_arrows(vertices: np.ndarray, faces: np.ndarray, 
                     arrows: np.ndarray, arrow_size: float, mvp: np.ndarray) -> Tuple:
    """Transform arrow positions and directions."""
    # Calculate arrow base positions (slightly above surface)
    vertex_normal_orig = vertex_normals(vertices, faces)
    A_base = np.c_[vertices + vertex_normal_orig * 0.01, np.ones(len(vertices))] @ mvp.T
    A_base /= A_base[:, 3].reshape(-1, 1)
    
    # Calculate arrow directions
    A_dir = np.copy(arrows)
    max_arrow = np.max(np.linalg.norm(arrows, axis=1))
    A_dir = arrow_size * A_dir / max_arrow
    A_dir = np.c_[A_dir, np.ones(len(A_dir))] @ mvp.T  
    A_dir /= A_dir[:, 3].reshape(-1, 1)
    
    return A_base, A_dir


def _render_view(vertices: np.ndarray, faces: np.ndarray, colours: np.ndarray,
                intensity: np.ndarray, view_angle: float, x_rotate: float, 
                z_rotate: float, flat_map: bool, show_back: bool, cmap: str,
                transparency: float, plane_intersections: Optional[Dict],
                plane_colours: Optional[np.ndarray], arrows: Optional[np.ndarray],
                arrow_subset: Optional[np.ndarray], arrow_size: float,
                arrow_colours: Optional[List], arrow_head: float, arrow_width: float,
                ax) -> PolyCollection:
    """Render a single view of the surface."""
    # Create transformation matrix
    mvp = create_mvp_matrix(view_angle, z_rotate, x_rotate, flat_map)
    
    # Transform vertices
    V = np.c_[vertices, np.ones(len(vertices))] @ mvp.T
    V /= V[:, 3].reshape(-1, 1)
    
    # Transform plane if present
    if plane_intersections is not None:
        P = np.c_[plane_intersections['coords'], 
                  np.ones(len(plane_intersections['coords']))] @ mvp.T
        P /= P[:, 3].reshape(-1, 1)
    
    # Get center for arrow depth testing
    center = np.array([0, 0, 0, 1]) @ mvp.T
    center /= center[3]
    
    # Handle arrows transformation
    if arrows is not None:
        A_base, A_dir = _transform_arrows(vertices, faces, arrows, arrow_size, mvp)
    
    # Get triangle coordinates and depths
    V_faces = V[faces]
    T = V_faces[:, :, :2]  # 2D triangle coordinates
    Z = -V_faces[:, :, 2].mean(axis=1)  # Depth values
    
    # Apply intensity shading
    shaded_colours = apply_intensity_shading(colours, intensity)
    
    # Handle front/back culling
    front, _ = frontback(T)
    if not show_back:
        T = T[front]
        shaded_colours = shaded_colours[front]
        Z = Z[front]
        if plane_intersections is not None:
            plane_masks = {
                'behind': plane_intersections['behind'][front],
                'intersected': plane_intersections['intersected'][front],
                'in_front': plane_intersections['in_front'][front]
            }
        else:
            plane_masks = None
    else:
        plane_masks = plane_intersections
    
    # Render with or without plane
    collection = None
    if plane_intersections is not None and plane_masks is not None:
        collection = _render_with_plane(ax, T, shaded_colours, Z, plane_masks, P, 
                                      plane_colours, cmap, transparency)
    else:
        collection = _render_simple(ax, T, shaded_colours, Z, cmap, transparency)
    
    # Add arrows
    if arrows is not None:
        _add_arrows(ax, A_base, A_dir, arrow_subset, faces, front, center,
                   arrow_colours, arrow_head, arrow_width)
    
    return collection


def _render_with_plane(ax, triangles: np.ndarray, colours: np.ndarray, 
                      depths: np.ndarray, plane_masks: Dict, plane_coords: np.ndarray,
                      plane_colours: np.ndarray, cmap: str, transparency: float) -> PolyCollection:
    """Render surface with plane intersections."""
    if np.sum(plane_masks['intersected']) > 0:
        # Sort rendering order based on depth
        if (np.median(depths[plane_masks['behind']]) > 
            np.median(depths[plane_masks['in_front']])):
            # Swap if behind faces are actually in front
            plane_masks['behind'], plane_masks['in_front'] = \
                plane_masks['in_front'], plane_masks['behind']
        
        # Render in depth order: behind -> intersected -> plane -> in_front
        for mask_name in ['behind', 'intersected']:
            mask = plane_masks[mask_name]
            if np.any(mask):
                indices = np.argsort(depths[mask])
                collection = PolyCollection(
                    triangles[mask][indices], closed=True, linewidth=0,
                    antialiased=False, facecolor=colours[mask][indices], cmap=cmap)
                collection.set_alpha(transparency)
                ax.add_collection(collection)
        
        # Render plane
        plane_faces = np.array([[0, 1, 3, 2]])
        PT = plane_coords[plane_faces][:, :, :2]
        collection = PolyCollection(
            PT, closed=True, linewidth=0, antialiased=True,
            facecolor=plane_colours, cmap=cmap)
        collection.set_alpha(plane_colours[:, 3].mean())
        ax.add_collection(collection)
        
        # Render in-front faces
        mask = plane_masks['in_front']
        if np.any(mask):
            indices = np.argsort(depths[mask])
            collection = PolyCollection(
                triangles[mask][indices], closed=True, linewidth=0,
                antialiased=False, facecolor=colours[mask][indices], cmap=cmap)
            collection.set_alpha(transparency)
            ax.add_collection(collection)
    
    return collection


def _render_simple(ax, triangles: np.ndarray, colours: np.ndarray, 
                  depths: np.ndarray, cmap: str, transparency: float) -> PolyCollection:
    """Render surface without plane intersections."""
    indices = np.argsort(depths)
    collection = PolyCollection(
        triangles[indices], closed=True, linewidth=0, 
        antialiased=False, facecolor=colours[indices], cmap=cmap)
    collection.set_alpha(transparency)
    ax.add_collection(collection)
    return collection


def _add_arrows(ax, A_base: np.ndarray, A_dir: np.ndarray, 
               arrow_subset: np.ndarray, faces: np.ndarray, front_mask: np.ndarray,
               center: np.ndarray, arrow_colours: Optional[List], 
               arrow_head: float, arrow_width: float) -> None:
    """Add arrow overlays to the plot."""
    front_arrows = faces[front_mask].ravel()
    
    for arrow_index, i in enumerate(arrow_subset):
        if i in front_arrows and A_base[i, 2] < center[2] + 0.01:
            arrow_colour = 'k'
            if arrow_colours is not None:
                arrow_colour = arrow_colours[arrow_index]
            
            # Handle different arrow direction array sizes
            if len(A_dir) == len(A_base):
                direction = A_dir[i]
            elif len(A_dir) == len(arrow_subset):
                direction = A_dir[arrow_index]
            
            half = direction * 0.5
            ax.arrow(A_base[i, 0] - half[0], A_base[i, 1] - half[1],
                    direction[0], direction[1], head_width=arrow_head,
                    width=arrow_width, color=arrow_colour)


def plot_surf(vertices, faces, overlay, rotate=[90, 270], cmap='viridis', 
             filename='plot.png', label=False, vmax=None, vmin=None, 
             x_rotate=270, pvals=None, colorbar=True, cmap_label='value',
             title=None, mask=None, base_size=6, arrows=None, arrow_subset=None,
             arrow_size=0.5, arrow_colours=None, arrow_head=0.05, 
             arrow_width=0.001, mask_colour=None, transparency=1, 
             show_back=False, border_colour=np.array([1, 0, 0, 1]),
             alpha_colour=None, flat_map=False, z_rotate=0, neighbours=None,
             parcel=None, parcel_cmap=None, filled_parcels=False, 
             return_ax=False, plane=None, random_seed=None):
    """Plot mesh surface with overlay data.
    
    This function provides comprehensive 3D surface visualization with support for:
    - Multiple viewing angles and flat map projection
    - Statistical overlays with p-value highlighting  
    - Parcellation delineation and filled regions
    - Vector field arrows and gradient visualization
    - Anatomical plane intersections
    - Transparency and masking effects
    
    Parameters
    ----------
    vertices : array_like, shape (n_vertices, 3)
        3D coordinates of mesh vertices
    faces : array_like, shape (n_faces, 3) 
        Triangle connectivity defining mesh topology
    overlay : array_like or list of array_like
        Data values to visualize on surface
    rotate : list of float, default [90, 270]
        Viewing angles for surface rotation
    cmap : str, default 'viridis'
        Matplotlib colormap name
    filename : str, optional
        Output filename for saving figure
    label : bool, default False
        Use median (True) vs mean (False) for face coloring
    vmin, vmax : float, optional
        Value range for color mapping
    x_rotate : float, default 270
        X-axis rotation angle in degrees
    pvals : array_like, optional
        P-values for statistical highlighting
    colorbar : bool, default True
        Whether to display colorbar
    cmap_label : str, default 'value'
        Colorbar label text
    title : str, optional
        Figure title
    mask : array_like, optional
        Boolean mask for hiding surface regions
    base_size : int, default 6
        Base figure size scaling factor
    arrows : array_like, optional
        Vector field directions for arrow overlay
    arrow_subset : array_like, optional
        Vertex indices where arrows should be displayed
    arrow_size : float, default 0.5
        Scaling factor for arrow lengths
    arrow_colours : list, optional
        Colors for individual arrows
    arrow_head : float, default 0.05
        Arrow head width
    arrow_width : float, default 0.001
        Arrow shaft width
    mask_colour : array_like, optional
        RGBA color for masked regions
    transparency : float, default 1
        Surface transparency (0-1)
    show_back : bool, default False
        Whether to render back-facing triangles
    border_colour : array_like, default [1,0,0,1]
        RGBA color for significant cluster borders
    alpha_colour : array_like, optional
        Per-vertex alpha values for transparency effects
    flat_map : bool, default False
        Use flat map projection instead of 3D
    z_rotate : float, default 0
        Z-axis rotation angle in degrees
    neighbours : array_like, optional
        Pre-computed vertex neighbor lists
    parcel : array_like, optional
        Parcellation labels for region delineation
    parcel_cmap : dict, optional
        Mapping from parcel IDs to RGBA colors
    filled_parcels : bool, default False
        Fill parcels with solid colors vs boundaries only
    return_ax : bool, default False
        Return figure, axis, and transformation matrix
    plane : dict, optional
        Plane intersection configuration with keys:
        - 'mri_img': MRI image for plane definition
        - 'slice_i': Slice index
        - 'slice_axis': Axis for slicing (0, 1, or 2)
        - 'plane_colour': RGBA plane color
        - 'plane_alpha': Plane transparency
        - 'intersected_colour': Color for intersected faces
    random_seed : int, optional
        Random seed for reproducible color generation (default: None)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object (if return_ax=True)
    ax : matplotlib.axes.Axes 
        Axes object (if return_ax=True)
    mvp : ndarray
        Model-view-projection matrix (if return_ax=True)
    """
    # Set random seed for reproducible results
    if random_seed is not None:
        set_random_seed(random_seed)
    
    # Setup plane configuration
    default_plane = {
        'mri_img': None, 'slice_i': 100, 'slice_axis': 1,
        'plane_colour': np.array([.5, .5, .5, 1]),
        'plane_alpha': 0.8,
        'intersected_colour': np.array([0, 0, 0, 1])
    }
    if plane is not None:
        if not isinstance(plane, dict):
            raise ValueError('Plane should be a dictionary with required keys')
        for key in default_plane.keys():
            if key not in plane:
                plane[key] = default_plane[key]
    
    # Input validation and normalization
    if not isinstance(rotate, list):
        rotate = [rotate]
    if not isinstance(overlay, list):
        overlays = [overlay]
    else:
        overlays = overlay
    if parcel is not None and parcel.sum() == 0:
        parcel = None
    
    # Handle flat map constraints
    if flat_map:
        z_rotate = 90
        rotate = [90]
        if plane is not None:
            print('Plane is not supported for flat maps, ignoring plane.')
            plane = None
    
    # Prepare surface data
    vertices, faces, overlays, plane_data = _prepare_surface_data(
        vertices, faces, overlays, plane)
    
    # Setup lighting and plane intersections
    intensity, plane_intersections, plane_colours = _setup_lighting_and_plane(
        vertices, faces, plane_data, z_rotate, flat_map, plane)
    
    # Create figure
    fig = plt.figure(figsize=(base_size * len(rotate) + colorbar * (base_size - 2),
                              (base_size - 1) * len(overlays)))
    if title is not None:
        plt.title(title, fontsize=25)
    plt.axis('off')
    
    # Process each overlay
    collection = None
    for k, overlay in enumerate(overlays):
        # Process overlay colors
        colours, vmin_used, vmax_used = _process_overlay_colours(
            overlay, faces, cmap, vmin, vmax, label, alpha_colour, pvals,
            mask, mask_colour, border_colour, parcel, parcel_cmap,
            filled_parcels, neighbours, plane_intersections, plane)
        
        # Render each view
        for i, view_angle in enumerate(rotate):
            ax = fig.add_subplot(len(overlays), len(rotate) + 1, 2 * k + i + 1,
                               xlim=[-.98, +.98], ylim=[-.98, +.98], aspect=1,
                               frameon=False, xticks=[], yticks=[])
            
            collection = _render_view(
                vertices, faces, colours, intensity, view_angle, x_rotate,
                z_rotate, flat_map, show_back, cmap, transparency,
                plane_intersections, plane_colours, arrows, arrow_subset,
                arrow_size, arrow_colours, arrow_head, arrow_width, ax)
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # Add colorbar
    if colorbar and collection is not None:
        l = 0.7 if len(rotate) > 1 else 0.5
        cbar_size = [l, 0.3, 0.03, 0.38]
        cbar = fig.colorbar(collection, ticks=[0, 0.5, 1],
                           cax=fig.add_axes(cbar_size))
        cbar.ax.set_yticklabels([
            np.round(vmin_used, decimals=2),
            np.round(np.mean([vmin_used, vmax_used]), decimals=2),
            np.round(vmax_used, decimals=2)
        ])
        cbar.ax.tick_params(labelsize=25)
        cbar.ax.set_title(cmap_label, fontsize=25, pad=30)
    
    # Save figure
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)
    
    # Return results
    if return_ax:
        mvp = create_mvp_matrix(rotate[0], z_rotate, x_rotate, flat_map)
        return fig, ax, mvp
    return

