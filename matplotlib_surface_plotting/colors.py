"""Color manipulation utilities for surface plotting."""

import numpy as np
from typing import Optional, Dict, Any
from .geometry import get_neighbours_from_tris, get_ring_of_neighbours


def mask_colours(colours: np.ndarray, triangles: np.ndarray, 
                mask: Optional[np.ndarray], 
                mask_colour: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply mask to colours by greying out masked regions.
    
    Args:
        colours: Face colours array (n, 4)
        triangles: Triangle indices array (m, 3)
        mask: Boolean mask for vertices
        mask_colour: RGBA colour for masked regions
        
    Returns:
        Modified colours array
    """
    if mask is not None:
        if mask_colour is None:
            mask_colour = np.array([0.86, 0.86, 0.86, 1])
        verts_masked = mask[triangles].any(axis=1)
        colours[verts_masked, :] = mask_colour
    return colours


def adjust_colours_pvals(colours: np.ndarray, pvals: np.ndarray, 
                        triangles: np.ndarray, mask: Optional[np.ndarray] = None,
                        mask_colour: Optional[np.ndarray] = None,
                        border_colour: np.ndarray = np.array([1.0, 0, 0, 1])) -> np.ndarray:
    """Add red ring around significant clusters and grey out non-significant vertices.
    
    Args:
        colours: Face colours array (n, 4)
        pvals: P-values for each vertex
        triangles: Triangle indices array (m, 3)
        mask: Optional mask for vertices
        mask_colour: RGBA colour for masked regions
        border_colour: RGBA colour for cluster borders
        
    Returns:
        Modified colours array with significance highlighting
    """
    colours = mask_colours(colours, triangles, mask, mask_colour)
    neighbours = get_neighbours_from_tris(triangles)
    ring = get_ring_of_neighbours(pvals < 0.05, neighbours)
    
    if len(ring) > 0:
        ring_label = np.zeros(len(neighbours)).astype(bool)
        ring_label[ring] = 1
        ring = get_ring_of_neighbours(ring_label, neighbours)
        ring_label[ring] = 1
        colours[ring_label[triangles].any(axis=1), :] = border_colour
    
    grey_out = pvals < 0.05
    verts_grey_out = grey_out[triangles].any(axis=1)
    colours[verts_grey_out, :] = (1.5 * colours[verts_grey_out] + 
                                  np.array([0.86, 0.86, 0.86, 1])) / 2.5
    return colours


def add_parcellation_colours(colours: np.ndarray, parcel: np.ndarray, 
                           triangles: np.ndarray, labels: Optional[Dict] = None,
                           mask: Optional[np.ndarray] = None, filled: bool = False,
                           mask_colour: Optional[np.ndarray] = None, 
                           neighbours: Optional[np.ndarray] = None) -> np.ndarray:
    """Add parcellation delineation to surface colours.
    
    Args:
        colours: Face colours array (n, 4)
        parcel: Parcellation labels for each vertex
        triangles: Triangle indices array (m, 3)  
        labels: Dictionary mapping parcel IDs to RGBA colours
        mask: Optional mask for vertices
        filled: Whether to fill parcels with solid colours
        mask_colour: RGBA colour for masked regions
        neighbours: Pre-computed neighbour lists
        
    Returns:
        Modified colours array with parcellation
    """
    colours = mask_colours(colours, triangles, mask, mask_colour=mask_colour)
    
    # Normalize ROIs and colors
    rois = list(set(parcel))
    if 0 in rois:
        rois.remove(0)
    
    if labels is None:
        labels = dict(zip(rois, np.random.rand(len(rois), 4)))
    
    # Fill parcels with solid colours
    if filled:
        colours = np.zeros_like(colours)
        for label in rois:
            colours[np.median(parcel[triangles], axis=1) == label] = labels[label]
        return colours
    
    # Create parcel boundaries
    if neighbours is None:
        neighbours = get_neighbours_from_tris(triangles)
    
    matrix_colored = np.zeros([len(triangles), len(rois)])
    for l, label in enumerate(rois):
        ring = get_ring_of_neighbours(parcel != label, neighbours)
        if len(ring) > 0:
            ring_label = np.zeros(len(neighbours)).astype(bool)
            ring_label[ring] = 1
            matrix_colored[:, l] = np.median(ring_label[triangles], axis=1)
    
    # Update colours with delineation
    maxis = [max(matrix_colored[i, :]) for i in range(len(colours))]
    colours = np.array([
        labels[rois[np.random.choice(np.where(matrix_colored[i, :] == maxi)[0])]] 
        if maxi != 0 else colours[i] 
        for i, maxi in enumerate(maxis)
    ])
    return colours


def adjust_colours_alpha(colours: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Apply alpha transparency to colours based on scalar values.
    
    Args:
        colours: Face colours array (n, 4)
        alpha: Alpha values for each face
        
    Returns:
        Modified colours array with alpha blending
    """
    # Rescale alpha to 0.1-1.0 range
    alpha_rescaled = 0.1 + 0.9 * (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
    
    # Blend with grey background
    grey_bg = np.array([0.86, 0.86, 0.86, 1])
    colours = ((alpha_rescaled * colours.T).T + 
               ((1 - alpha_rescaled) * grey_bg.reshape(-1, 1)).T)
    
    return np.clip(colours, 0, 1)


def apply_intensity_shading(colours: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Apply lighting intensity to RGB channels of colours.
    
    Args:
        colours: Face colours array (n, 4)
        intensity: Lighting intensity for each face
        
    Returns:
        Colours with intensity applied to RGB channels
    """
    colours = colours.copy()
    colours[:, 0] *= intensity
    colours[:, 1] *= intensity  
    colours[:, 2] *= intensity
    return colours


def create_colormap_colours(overlay: np.ndarray, faces: np.ndarray, cmap: str,
                          vmin: Optional[float] = None, vmax: Optional[float] = None,
                          label: bool = False) -> tuple:
    """Create colours from overlay data using matplotlib colormap.
    
    Args:
        overlay: Data values to map to colours
        faces: Triangle indices array
        cmap: Matplotlib colormap name
        vmin, vmax: Value range for colour mapping
        label: Whether to use median (True) or mean (False) for face values
        
    Returns:
        Tuple of (colours_array, vmin_used, vmax_used)
    """
    import matplotlib.pyplot as plt
    
    # Calculate face values from vertex values
    if label:
        face_values = np.median(overlay[faces], axis=1)
    else:
        face_values = np.mean(overlay[faces], axis=1)
    
    # Normalize values
    if vmax is not None:
        normalized_values = (face_values - vmin) / (vmax - vmin)
        normalized_values = np.clip(normalized_values, 0, 1)
        vmin_used, vmax_used = vmin, vmax
    else:
        vmax_used = face_values.max()
        vmin_used = face_values.min()
        normalized_values = ((face_values - vmin_used) / 
                           (vmax_used - vmin_used))
    
    # Apply colormap
    colours = plt.get_cmap(cmap)(normalized_values)
    
    return colours, vmin_used, vmax_used