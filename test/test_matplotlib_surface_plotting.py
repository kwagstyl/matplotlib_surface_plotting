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
    
    @pytest.fixture
    def setup_gii_data(self):
        """Load test data from GIFTI files for examples."""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
        # Load inflated surface
        surf_inflated = nb.load(os.path.join(data_dir, 'fs_LR.32k.L.inflated.surf.gii'))
        vertices_inflated = surf_inflated.darrays[0].data
        faces_inflated = surf_inflated.darrays[1].data
        
        # Load flat surface
        surf_flat = nb.load(os.path.join(data_dir, 'fs_LR.32k.L.flat.surf.gii'))
        vertices_flat = surf_flat.darrays[0].data
        faces_flat = surf_flat.darrays[1].data
        
        # Load overlay data
        overlay_myelin = nb.load(os.path.join(data_dir, 'S1200.MyelinMap.L.func.gii')).darrays[0].data
        overlay_v1 = nb.load(os.path.join(data_dir, 'v1_geodesic.func.gii')).darrays[0].data
        
        # Load parcellation data
        parcellation = nb.load(os.path.join(data_dir, 'Glasser_2016.32k.L.label.gii')).darrays[0].data
        cortex_label = np.where(parcellation > 0)[0]
        
        # Load vector data for arrows
        d_inflated = nb.load(os.path.join(data_dir, 'geodesic_distance_inflated_vectors.func.gii'))
        arrows_inflated = np.vstack([d_inflated.darrays[0].data, d_inflated.darrays[1].data, d_inflated.darrays[2].data]).T
        
        d_flat = nb.load(os.path.join(data_dir, 'geodesic_distance_flat_vectors.func.gii'))
        arrows_flat = np.vstack([d_flat.darrays[0].data, d_flat.darrays[1].data, d_flat.darrays[2].data]).T
        
        return {
            'vertices_inflated': vertices_inflated,
            'faces_inflated': faces_inflated,
            'vertices_flat': vertices_flat,
            'faces_flat': faces_flat,
            'overlay_myelin': overlay_myelin,
            'overlay_v1': overlay_v1,
            'parcellation': parcellation,
            'cortex_label': cortex_label,
            'arrows_inflated': arrows_inflated,
            'arrows_flat': arrows_flat
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
    
    def _compare_images(self, generated_path, reference_path, max_mse=50):
        """Helper method to compare two images using MSE."""
        assert os.path.exists(reference_path), f"Reference image not found: {reference_path}"
        assert os.path.exists(generated_path), f"Generated image not found: {generated_path}"
        
        with Image.open(reference_path) as ref_img, Image.open(generated_path) as test_img:
            assert ref_img.size == test_img.size, f"Image dimensions don't match: {ref_img.size} vs {test_img.size}"
            
            # Convert to numpy arrays for comparison
            ref_array = np.array(ref_img)
            test_array = np.array(test_img)
            
            # Calculate mean squared error between images
            mse = np.mean((ref_array - test_array) ** 2)
            
            assert mse < max_mse, f"Images differ too much (MSE: {mse})"
    
    def test_alpha_colours_example(self, setup_gii_data):
        """Test alpha colours example from notebook."""
        data = setup_gii_data
        
        # Recreate alpha calculation from notebook
        alpha = np.cos(data['vertices_inflated'][:,1]/10) + np.sin(data['vertices_inflated'][:,2]/10)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_output = tmp_file.name
        
        try:
            plot_surf(
                data['vertices_inflated'], 
                data['faces_inflated'], 
                data['overlay_myelin'],
                alpha_colour=alpha,
                flat_map=False,
                vmin=1,
                vmax=2,
                filename=tmp_output
            )
            
            reference_path = os.path.join(os.path.dirname(__file__), "reference_alpha_colours.png")
            self._compare_images(tmp_output, reference_path)
            
        finally:
            if os.path.exists(tmp_output):
                os.unlink(tmp_output)
    
    def test_arrows_map_inflated_example(self, setup_gii_data):
        """Test arrows map on inflated surface example from notebook."""
        data = setup_gii_data
        
        # Choose subset of arrows as in notebook
        np.random.seed(42)  # For reproducible results
        selection = np.random.choice(data['cortex_label'], 500)
        
        # Create arrow colours
        import matplotlib.cm as cm
        cmap = cm.get_cmap('viridis')
        arrow_colours = cmap(selection/np.max(selection))
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_output = tmp_file.name
        
        try:
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
                filename=tmp_output
            )
            
            reference_path = os.path.join(os.path.dirname(__file__), "reference_arrows_inflated.png")
            self._compare_images(tmp_output, reference_path)
            
        finally:
            if os.path.exists(tmp_output):
                os.unlink(tmp_output)
    
    def test_arrows_map_flat_example(self, setup_gii_data):
        """Test arrows map on flat surface example from notebook."""
        data = setup_gii_data
        
        # Choose subset of arrows as in notebook
        np.random.seed(42)  # For reproducible results
        selection = np.random.choice(data['cortex_label'], 500)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_output = tmp_file.name
        
        try:
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
                filename=tmp_output
            )
            
            reference_path = os.path.join(os.path.dirname(__file__), "reference_arrows_flat.png")
            self._compare_images(tmp_output, reference_path)
            
        finally:
            if os.path.exists(tmp_output):
                os.unlink(tmp_output)
    
    def test_flat_map_example(self, setup_gii_data):
        """Test flat map example from notebook."""
        data = setup_gii_data
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_output = tmp_file.name
        
        try:
            plot_surf(
                data['vertices_flat'],
                data['faces_flat'],
                data['overlay_myelin'],
                flat_map=True,
                base_size=10,
                vmin=1,
                vmax=2,
                filename=tmp_output
            )
            
            reference_path = os.path.join(os.path.dirname(__file__), "reference_flat_map.png")
            self._compare_images(tmp_output, reference_path)
            
        finally:
            if os.path.exists(tmp_output):
                os.unlink(tmp_output)
    
    def test_parcellation_example(self, setup_data):
        """Test parcellation example from notebook."""
        data = setup_data
        
        # Load parcellation as in notebook
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        atlas = nb.freesurfer.io.read_annot(os.path.join(data_dir, 'lh.aparc.annot'))[0]
        rois = list(set(atlas))
        
        # Create reproducible colors
        np.random.seed(42)
        colors = np.random.rand(len(rois), 4)
        label_atlas = dict(zip(rois, colors))
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_output = tmp_file.name
        
        try:
            plot_surf(
                data['vertices'],
                data['faces'],
                data['overlay'],
                rotate=[90, 270],
                vmax=np.max(data['overlay'][data['cortex']]),
                vmin=np.min(data['overlay'][data['cortex']]),
                pvals=np.ones_like(data['overlay']),
                cmap_label='thickness \\n(mm)',
                parcel=atlas,
                parcel_cmap=label_atlas,
                filename=tmp_output
            )
            
            reference_path = os.path.join(os.path.dirname(__file__), "reference_parcellation.png")
            self._compare_images(tmp_output, reference_path)
            
        finally:
            if os.path.exists(tmp_output):
                os.unlink(tmp_output)
