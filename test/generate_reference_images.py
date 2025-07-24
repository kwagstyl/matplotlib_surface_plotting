#!/usr/bin/env python3
"""
Script to generate reference images for test cases.
Run this script to create the reference images that tests will compare against.
"""

import os
import sys
import tempfile
import numpy as np
import nibabel as nb
import matplotlib.cm as cm

# Add the parent directory to the path to import matplotlib_surface_plotting
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib_surface_plotting import plot_surf

def setup_data():
    """Load all required data for generating reference images."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Load FreeSurfer data
    vertices_fs, faces_fs = nb.freesurfer.io.read_geometry(os.path.join(data_dir, 'lh.inflated'))
    overlay_fs = nb.freesurfer.io.read_morph_data(os.path.join(data_dir, 'lh.thickness'))
    cortex = nb.freesurfer.io.read_label(os.path.join(data_dir, 'lh.cortex.label'))
    mask = np.ones_like(overlay_fs).astype(bool)
    mask[cortex] = 0
    overlay_fs[mask] = np.min(overlay_fs)
    
    # Load GIFTI data
    surf_inflated = nb.load(os.path.join(data_dir, 'fs_LR.32k.L.inflated.surf.gii'))
    vertices_inflated = surf_inflated.darrays[0].data
    faces_inflated = surf_inflated.darrays[1].data
    
    surf_flat = nb.load(os.path.join(data_dir, 'fs_LR.32k.L.flat.surf.gii'))
    vertices_flat = surf_flat.darrays[0].data
    faces_flat = surf_flat.darrays[1].data
    
    overlay_myelin = nb.load(os.path.join(data_dir, 'S1200.MyelinMap.L.func.gii')).darrays[0].data
    overlay_v1 = nb.load(os.path.join(data_dir, 'v1_geodesic.func.gii')).darrays[0].data
    
    parcellation = nb.load(os.path.join(data_dir, 'Glasser_2016.32k.L.label.gii')).darrays[0].data
    cortex_label = np.where(parcellation > 0)[0]
    
    # Load vector data
    d_inflated = nb.load(os.path.join(data_dir, 'geodesic_distance_inflated_vectors.func.gii'))
    arrows_inflated = np.vstack([d_inflated.darrays[0].data, d_inflated.darrays[1].data, d_inflated.darrays[2].data]).T
    
    d_flat = nb.load(os.path.join(data_dir, 'geodesic_distance_flat_vectors.func.gii'))
    arrows_flat = np.vstack([d_flat.darrays[0].data, d_flat.darrays[1].data, d_flat.darrays[2].data]).T
    
    # Load parcellation data
    atlas = nb.freesurfer.io.read_annot(os.path.join(data_dir, 'lh.aparc.annot'))[0]
    
    return {
        'vertices_fs': vertices_fs,
        'faces_fs': faces_fs,
        'overlay_fs': overlay_fs,
        'cortex': cortex,
        'mask': mask,
        'vertices_inflated': vertices_inflated,
        'faces_inflated': faces_inflated,
        'vertices_flat': vertices_flat,
        'faces_flat': faces_flat,
        'overlay_myelin': overlay_myelin,
        'overlay_v1': overlay_v1,
        'parcellation': parcellation,
        'cortex_label': cortex_label,
        'arrows_inflated': arrows_inflated,
        'arrows_flat': arrows_flat,
        'atlas': atlas
    }

def generate_alpha_colours_reference(data, output_path):
    """Generate reference image for alpha colours example."""
    alpha = np.cos(data['vertices_inflated'][:,1]/10) + np.sin(data['vertices_inflated'][:,2]/10)
    
    plot_surf(
        data['vertices_inflated'], 
        data['faces_inflated'], 
        data['overlay_myelin'],
        alpha_colour=alpha,
        flat_map=False,
        vmin=1,
        vmax=2,
        filename=output_path
    )
    print(f"Generated: {output_path}")

def generate_arrows_inflated_reference(data, output_path):
    """Generate reference image for arrows on inflated surface."""
    np.random.seed(42)  # For reproducible results
    selection = np.random.choice(data['cortex_label'], 500)
    
    cmap = cm.get_cmap('viridis')
    arrow_colours = cmap(selection/np.max(selection))
    
    plot_surf(
        data['vertices_inflated'],
        data['faces_inflated'],
        data['overlay_v1'],
        flat_map=False,
        base_size=10,
        vmin=1,
        vmax=200,
        arrows=data['arrows_inflated'],
        arrow_subset=selection,
        arrow_size=0.1,
        arrow_colours=arrow_colours,
        cmap='turbo',
        filename=output_path
    )
    print(f"Generated: {output_path}")

def generate_arrows_flat_reference(data, output_path):
    """Generate reference image for arrows on flat surface."""
    np.random.seed(42)  # For reproducible results
    selection = np.random.choice(data['cortex_label'], 500)
    
    plot_surf(
        data['vertices_flat'],
        data['faces_flat'],
        data['overlay_v1'],
        flat_map=True,
        base_size=10,
        vmin=1,
        vmax=200,
        arrows=data['arrows_flat'],
        arrow_subset=selection,
        arrow_size=0.8,
        cmap='turbo',
        filename=output_path
    )
    print(f"Generated: {output_path}")

def generate_flat_map_reference(data, output_path):
    """Generate reference image for flat map example."""
    plot_surf(
        data['vertices_flat'],
        data['faces_flat'],
        data['overlay_myelin'],
        flat_map=True,
        base_size=10,
        vmin=1,
        vmax=2,
        filename=output_path
    )
    print(f"Generated: {output_path}")

def generate_parcellation_reference(data, output_path):
    """Generate reference image for parcellation example."""
    rois = list(set(data['atlas']))
    
    # Create reproducible colors
    np.random.seed(42)
    colors = np.random.rand(len(rois), 4)
    label_atlas = dict(zip(rois, colors))
    
    plot_surf(
        data['vertices_fs'],
        data['faces_fs'],
        data['overlay_fs'],
        rotate=[90, 270],
        vmax=np.max(data['overlay_fs'][data['cortex']]),
        vmin=np.min(data['overlay_fs'][data['cortex']]),
        pvals=np.ones_like(data['overlay_fs']),
        cmap_label='thickness \\n(mm)',
        parcel=data['atlas'],
        parcel_cmap=label_atlas,
        filename=output_path
    )
    print(f"Generated: {output_path}")

def main():
    """Generate all reference images."""
    print("Loading data...")
    data = setup_data()
    
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test')
    
    print("Generating reference images...")
    
    generate_alpha_colours_reference(
        data, 
        os.path.join(test_dir, 'reference_alpha_colours.png')
    )
    
    generate_arrows_inflated_reference(
        data,
        os.path.join(test_dir, 'reference_arrows_inflated.png')
    )
    
    generate_arrows_flat_reference(
        data,
        os.path.join(test_dir, 'reference_arrows_flat.png')
    )
    
    generate_flat_map_reference(
        data,
        os.path.join(test_dir, 'reference_flat_map.png')
    )
    
    generate_parcellation_reference(
        data,
        os.path.join(test_dir, 'reference_parcellation.png')
    )
    
    print("All reference images generated successfully!")

if __name__ == "__main__":
    main()