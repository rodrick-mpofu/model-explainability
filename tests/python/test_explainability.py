"""
Test suite for the explainability API.

This module contains tests for the main explainability functionality
including model loading and explanation generation.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/python'))

try:
    from explainability_api import explain_image
    from models.model_loader import load_model, get_imagenet_label
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "TensorFlow dependencies not available")
class TestExplainabilityAPI(unittest.TestCase):
    """Test cases for the explainability API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image_path = "assets/images/test_image.jpg"
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_model_vgg16(self):
        """Test loading VGG16 model."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")
        
        model, last_conv_layer, preprocess, input_size = load_model("vgg16")
        self.assertIsNotNone(model)
        self.assertEqual(last_conv_layer, "block5_conv3")
        self.assertEqual(input_size, (224, 224))
    
    def test_load_model_invalid(self):
        """Test loading invalid model raises error."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")
        
        with self.assertRaises(ValueError):
            load_model("invalid_model")
    
    def test_get_imagenet_label(self):
        """Test ImageNet label retrieval."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")
        
        # Test with a known label index (e.g., index 1 is goldfish)
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"1": ["n01443537", "goldfish"]}
            mock_get.return_value = mock_response
            
            label = get_imagenet_label(1)
            self.assertEqual(label, "goldfish")
    
    @patch('os.path.exists')
    def test_explain_image_file_not_found(self, mock_exists):
        """Test that FileNotFoundError is raised for non-existent image."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")
        
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            explain_image("non_existent_image.jpg")
    
    def test_explain_image_invalid_technique(self):
        """Test that ValueError is raised for invalid technique."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")
        
        with patch('os.path.exists', return_value=True):
            with self.assertRaises(ValueError):
                explain_image("test.jpg", technique="invalid_technique")


class TestBackwardCompatibility(unittest.TestCase):
    """Test cases for backward compatibility functions."""
    
    def test_import_compatibility(self):
        """Test that old import paths still work."""
        # This test ensures that the reorganization doesn't break existing code
        try:
            # These should work if the old files are still referenced correctly
            from explainability_api import generate_gradcam_explanation
            from explainability_api import generate_shap_explanation
            self.assertTrue(callable(generate_gradcam_explanation))
            self.assertTrue(callable(generate_shap_explanation))
        except ImportError:
            # If dependencies aren't available, that's expected
            if not DEPENDENCIES_AVAILABLE:
                self.skipTest("Dependencies not available")
            else:
                raise


if __name__ == '__main__':
    unittest.main()