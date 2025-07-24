import os
import tempfile
import pytest
import numpy as np
import nibabel as nb
from matplotlib_surface_plotting import plot_surf
from PIL import Image


class TestSurfacePlotting:
    @pytest.fixture
    def setup_data(self):
        """Load test data for surface plotting."""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
        vertices, faces = nb.freesurfer.io.read_geometry(os.path.join(data_dir, 'lh.inflated'))
        overlay = nb.freesurfer.io.read_morph_data(os.path.join(data_dir, 'lh.thickness'))
        
        # Optional masking of medial wall
        cortex = nb.freesurfer.io.read_label(os.path.join(data_dir, 'lh.cortex.label'))
        mask = np.ones_like(overlay).astype(bool)
        mask[cortex] = 0
        overlay[mask] = np.min(overlay)
        
        return {
            'vertices': vertices,
            'faces': faces,
            'overlay': overlay,
            'cortex': cortex,
            'mask': mask
        }
    
    def test_plot_surf_basic(self, setup_data):
        """Test basic surface plotting functionality."""
        data = setup_data
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_output = tmp_file.name
        
        try:
            plot_surf(
                data['vertices'], 
                data['faces'], 
                data['overlay'], 
                rotate=[90, 270], 
                filename=tmp_output,
                vmax=np.max(data['overlay'][data['cortex']]),
                vmin=np.min(data['overlay'][data['cortex']]),
                mask=data['mask'],
                pvals=np.ones_like(data['overlay']), 
                cmap_label='thickness \n(mm)'
            )
            
            # Check that the output file was created
            assert os.path.exists(tmp_output)
            
            # Load reference image and generated image for comparison
            reference_path = os.path.join(os.path.dirname(__file__), "test_plot.png")
            assert os.path.exists(reference_path), "Reference image not found"
            
            # Compare images - check dimensions and pixel data
            with Image.open(reference_path) as ref_img, Image.open(tmp_output) as test_img:
                assert ref_img.size == test_img.size, "Generated image dimensions don't match reference"
                
                # Convert to numpy arrays for comparison
                ref_array = np.array(ref_img)
                test_array = np.array(test_img)
                
                # Calculate mean squared error between images
                mse = np.mean((ref_array - test_array) ** 2)
                
                # Allow for small differences due to rendering variations
                assert mse < 50, f"Images differ too much (MSE: {mse})"
                
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_output):
                os.unlink(tmp_output)
    
    def test_data_shapes(self, setup_data):
        """Test that loaded data has expected shapes."""
        data = setup_data
        
        # Vertices should be Nx3 array
        assert data['vertices'].ndim == 2
        assert data['vertices'].shape[1] == 3
        
        # Faces should be Mx3 array  
        assert data['faces'].ndim == 2
        assert data['faces'].shape[1] == 3
        
        # Overlay should be 1D array matching number of vertices
        assert data['overlay'].ndim == 1
        assert len(data['overlay']) == len(data['vertices'])
        
        # Mask should have same length as overlay
        assert len(data['mask']) == len(data['overlay'])
