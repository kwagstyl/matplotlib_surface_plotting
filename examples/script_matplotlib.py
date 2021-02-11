import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib_surface_plotting import plot_surf
import nibabel as nb
import numpy as np

vertices, faces=nb.freesurfer.io.read_geometry('../data/lh.inflated')
overlay = nb.freesurfer.io.read_morph_data('../data/lh.thickness')

#optional masking of medial wall
cortex=nb.freesurfer.io.read_label('../data/lh.cortex.label')
mask=np.ones_like(overlay).astype(bool)
mask[cortex]=0
overlay[mask]=np.min(overlay)

plot_surf( vertices, faces, overlay, rotate=[90,270], filename='demo_plot.png',
          vmax = np.max(overlay[cortex]),vmin=np.min(overlay[cortex]),mask=mask,
          pvals=np.ones_like(overlay), cmap_label='thickness \n(mm)')
