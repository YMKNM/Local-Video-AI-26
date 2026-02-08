"""
ONNX Model Loader - Handles model loading, conversion, and management

This module handles:
- Loading ONNX models from disk
- Converting PyTorch models to ONNX (if needed)
- Model caching and management
- Model validation
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import yaml

logger = logging.getLogger(__name__)


class ONNXModelLoader:
    """
    Manages ONNX model loading, caching, and conversion.
    
    Supports:
    - Loading pre-converted ONNX models
    - Converting PyTorch models to ONNX
    - Model validation
    - Caching for performance
    """
    
    def __init__(self, config_dir: Optional[Path] = None, models_dir: Optional[Path] = None):
        """
        Initialize the model loader.
        
        Args:
            config_dir: Path to configuration directory
            models_dir: Path to models directory
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / "models"
        
        self.config_dir = Path(config_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_config = self._load_model_config()
        self._loaded_models: Dict[str, Dict[str, Any]] = {}
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML"""
        config_path = self.config_dir / "models.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def get_model_path(self, model_type: str, model_name: str) -> Optional[Path]:
        """
        Get the path to a model's ONNX file.
        
        Args:
            model_type: Type of model (text_encoder, video_diffusion, vae, scheduler)
            model_name: Name of the specific model
            
        Returns:
            Path to ONNX model file, or None if not found
        """
        # Get model configuration
        type_config = self.model_config.get(model_type, {})
        
        # Handle plural forms
        if not type_config:
            type_config = self.model_config.get(f"{model_type}s", {})
        
        model_info = type_config.get(model_name, {})
        
        if not model_info:
            logger.warning(f"Model not found in config: {model_type}/{model_name}")
            return None
        
        # Get ONNX path from config
        onnx_path = model_info.get('onnx_path')
        
        if onnx_path:
            full_path = self.models_dir.parent / onnx_path
            if full_path.exists():
                return full_path
        
        # Fallback: look in standard location
        standard_path = self.models_dir / model_type / model_name / "model.onnx"
        if standard_path.exists():
            return standard_path
        
        return None
    
    def is_model_available(self, model_type: str, model_name: str) -> bool:
        """Check if a model is available locally"""
        path = self.get_model_path(model_type, model_name)
        return path is not None and path.exists()
    
    def list_available_models(self, model_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all available models.
        
        Args:
            model_type: Optional filter by type
            
        Returns:
            Dictionary of model_type -> list of available model names
        """
        available = {}
        
        model_types = [model_type] if model_type else [
            'text_encoders', 'video_diffusion', 'vae', 'schedulers'
        ]
        
        for mtype in model_types:
            type_config = self.model_config.get(mtype, {})
            available_models = []
            
            for model_name in type_config:
                # Check if ONNX file exists
                mtype_singular = mtype.rstrip('s') if mtype.endswith('s') else mtype
                if self.is_model_available(mtype_singular, model_name):
                    available_models.append(model_name)
            
            if available_models:
                available[mtype] = available_models
        
        return available
    
    def get_model_info(self, model_type: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a model from configuration"""
        type_config = self.model_config.get(model_type, {})
        if not type_config:
            type_config = self.model_config.get(f"{model_type}s", {})
        
        return type_config.get(model_name)
    
    def validate_onnx_model(self, model_path: Path) -> Tuple[bool, str]:
        """
        Validate an ONNX model file.
        
        Args:
            model_path: Path to ONNX file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            import onnx
            
            # Load and check model
            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)
            
            return True, ""
            
        except ImportError:
            # ONNX not installed, skip validation
            logger.warning("ONNX package not installed, skipping validation")
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    def get_model_metadata(self, model_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from an ONNX model.
        
        Args:
            model_path: Path to ONNX file
            
        Returns:
            Dictionary with model metadata
        """
        metadata = {
            'path': str(model_path),
            'size_mb': model_path.stat().st_size / (1024 * 1024),
        }
        
        try:
            import onnx
            model = onnx.load(str(model_path))
            
            # Get input/output info
            metadata['inputs'] = [
                {
                    'name': inp.name,
                    'shape': [d.dim_value if d.dim_value else d.dim_param 
                             for d in inp.type.tensor_type.shape.dim],
                    'dtype': onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
                }
                for inp in model.graph.input
            ]
            
            metadata['outputs'] = [
                {
                    'name': out.name,
                    'shape': [d.dim_value if d.dim_value else d.dim_param 
                             for d in out.type.tensor_type.shape.dim],
                    'dtype': onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
                }
                for out in model.graph.output
            ]
            
            # Get opset version
            metadata['opset_version'] = model.opset_import[0].version if model.opset_import else None
            
            # Get custom metadata
            for prop in model.metadata_props:
                metadata[f'custom_{prop.key}'] = prop.value
            
        except ImportError:
            logger.warning("ONNX package not installed, limited metadata available")
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
        
        return metadata
    
    def load_model(
        self,
        model_type: str,
        model_name: str,
        session_manager: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Load a model and create an inference session.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            session_manager: Optional DirectMLSession manager
            
        Returns:
            ONNX Runtime InferenceSession or None
        """
        model_path = self.get_model_path(model_type, model_name)
        
        if model_path is None:
            logger.error(f"Model not found: {model_type}/{model_name}")
            return None
        
        logger.info(f"Loading model: {model_path}")
        
        # Validate model
        is_valid, error = self.validate_onnx_model(model_path)
        if not is_valid:
            logger.error(f"Invalid ONNX model: {error}")
            return None
        
        # Create session
        if session_manager:
            session = session_manager.create_session(
                str(model_path),
                session_name=f"{model_type}_{model_name}"
            )
        else:
            # Create simple session without DirectML manager
            import onnxruntime as ort
            session = ort.InferenceSession(str(model_path))
        
        # Track loaded model
        self._loaded_models[f"{model_type}_{model_name}"] = {
            'path': model_path,
            'session': session,
            'metadata': self.get_model_metadata(model_path)
        }
        
        return session
    
    def unload_model(self, model_type: str, model_name: str):
        """Unload a model from memory"""
        key = f"{model_type}_{model_name}"
        if key in self._loaded_models:
            del self._loaded_models[key]
            logger.info(f"Unloaded model: {key}")
    
    def unload_all_models(self):
        """Unload all models from memory"""
        self._loaded_models.clear()
        logger.info("All models unloaded")
    
    def convert_pytorch_to_onnx(
        self,
        model: Any,
        output_path: Path,
        input_shapes: Dict[str, List[int]],
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> bool:
        """
        Convert a PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model
            output_path: Path to save ONNX model
            input_shapes: Dictionary of input name -> shape
            opset_version: ONNX opset version
            dynamic_axes: Optional dynamic axes specification
            
        Returns:
            True if successful
        """
        try:
            import torch
            
            # Create dummy inputs
            dummy_inputs = {}
            for name, shape in input_shapes.items():
                dummy_inputs[name] = torch.randn(*shape)
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                str(output_path),
                input_names=list(input_shapes.keys()),
                opset_version=opset_version,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True
            )
            
            logger.info(f"Model converted to ONNX: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert model: {e}")
            return False
    
    def download_model(
        self,
        model_type: str,
        model_name: str,
        source: Optional[str] = None
    ) -> bool:
        """
        Download a model from HuggingFace Hub.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            source: Optional HuggingFace model ID
            
        Returns:
            True if successful
        """
        model_info = self.get_model_info(model_type, model_name)
        if not model_info and not source:
            logger.error(f"Model not found in config: {model_type}/{model_name}")
            return False
        
        source = source or model_info.get('source')
        if not source:
            logger.error("No source specified for model download")
            return False
        
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            
            # Determine download path
            download_dir = self.models_dir / model_type / model_name
            download_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading model from: {source}")
            
            # Try to download specific ONNX file first
            try:
                hf_hub_download(
                    repo_id=source,
                    filename="model.onnx",
                    local_dir=download_dir
                )
            except Exception:
                # Download entire model directory
                snapshot_download(
                    repo_id=source,
                    local_dir=download_dir
                )
            
            logger.info(f"Model downloaded to: {download_dir}")
            return True
            
        except ImportError:
            logger.error(
                "huggingface_hub not installed. Install with: "
                "pip install huggingface_hub"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    
    def get_required_models(self) -> List[Dict[str, str]]:
        """Get list of models required by current configuration"""
        required = []
        
        active = self.model_config.get('active_models', {})
        
        for model_type, model_name in active.items():
            required.append({
                'type': model_type,
                'name': model_name,
                'available': self.is_model_available(model_type, model_name)
            })
        
        return required
    
    def check_all_models_available(self) -> Tuple[bool, List[str]]:
        """
        Check if all required models are available.
        
        Returns:
            Tuple of (all_available, list of missing models)
        """
        required = self.get_required_models()
        missing = [
            f"{m['type']}/{m['name']}"
            for m in required
            if not m['available']
        ]
        
        return len(missing) == 0, missing


class ModelRegistry:
    """
    Registry for tracking and managing multiple model loaders.
    
    Useful for complex pipelines with multiple model types.
    """
    
    def __init__(self):
        self.loaders: Dict[str, ONNXModelLoader] = {}
        self.default_loader: Optional[ONNXModelLoader] = None
    
    def register_loader(self, name: str, loader: ONNXModelLoader, default: bool = False):
        """Register a model loader"""
        self.loaders[name] = loader
        if default or self.default_loader is None:
            self.default_loader = loader
    
    def get_loader(self, name: Optional[str] = None) -> ONNXModelLoader:
        """Get a loader by name or default"""
        if name:
            return self.loaders.get(name)
        return self.default_loader
    
    def list_loaders(self) -> List[str]:
        """List registered loader names"""
        return list(self.loaders.keys())


def main():
    """Test model loader"""
    logging.basicConfig(level=logging.INFO)
    
    loader = ONNXModelLoader()
    
    print("\n=== ONNX Model Loader ===")
    print(f"Models directory: {loader.models_dir}")
    
    # Check required models
    all_available, missing = loader.check_all_models_available()
    print(f"\nAll models available: {all_available}")
    
    if missing:
        print("Missing models:")
        for m in missing:
            print(f"  - {m}")
    
    # List available models
    available = loader.list_available_models()
    print("\nAvailable models:")
    for mtype, models in available.items():
        print(f"  {mtype}:")
        for m in models:
            print(f"    - {m}")
    
    # Show required models
    print("\nRequired models:")
    for m in loader.get_required_models():
        status = "✓" if m['available'] else "✗"
        print(f"  {status} {m['type']}/{m['name']}")


if __name__ == "__main__":
    main()
