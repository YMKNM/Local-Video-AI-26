"""
Text Encoder Module

Handles text encoding for video generation using CLIP-based encoders.
Supports ONNX models for DirectML acceleration.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class TextEncoder:
    """
    Text encoder for video generation prompts.
    
    Supports:
    - T5-based encoders (for LTX-Video, CogVideoX)
    - CLIP-based encoders
    - ONNX format for DirectML acceleration
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "google/t5-v1_1-xxl",
        max_length: int = 256,
        device: str = "auto"
    ):
        """
        Initialize text encoder.
        
        Args:
            model_path: Path to ONNX model file
            model_name: HuggingFace model name (for tokenizer)
            max_length: Maximum sequence length
            device: Compute device ('auto', 'directml', 'cpu')
        """
        self.model_path = Path(model_path) if model_path else None
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        self._session = None
        self._tokenizer = None
        self._loaded = False
    
    def load(self) -> bool:
        """
        Load the text encoder model.
        
        Returns:
            True if loaded successfully
        """
        if self._loaded:
            return True
        
        try:
            # Load tokenizer
            self._load_tokenizer()
            
            # Load ONNX model if path provided
            if self.model_path and self.model_path.exists():
                self._load_onnx_model()
            else:
                logger.warning(f"Model path not found: {self.model_path}")
                # Use fallback placeholder encoder
                self._setup_placeholder()
            
            self._loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load text encoder: {e}")
            return False
    
    def _load_tokenizer(self):
        """Load the tokenizer"""
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=self.max_length
            )
            logger.info(f"Loaded tokenizer: {self.model_name}")
        except ImportError:
            logger.warning("transformers not installed, using basic tokenizer")
            self._tokenizer = BasicTokenizer(self.max_length)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}, using basic tokenizer")
            self._tokenizer = BasicTokenizer(self.max_length)
    
    def _load_onnx_model(self):
        """Load ONNX model for DirectML"""
        try:
            import onnxruntime as ort
            
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Select execution provider
            providers = self._get_providers()
            
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            logger.info(f"Loaded text encoder ONNX model: {self.model_path}")
            logger.info(f"Using provider: {self._session.get_providers()[0]}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self._setup_placeholder()
    
    def _get_providers(self) -> List[str]:
        """Get execution providers based on device setting"""
        if self.device == "cpu":
            return ['CPUExecutionProvider']
        
        # Try DirectML first for AMD GPUs
        providers = []
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            
            if 'DmlExecutionProvider' in available:
                providers.append('DmlExecutionProvider')
            
            providers.append('CPUExecutionProvider')
            
        except Exception:
            providers = ['CPUExecutionProvider']
        
        return providers
    
    def _setup_placeholder(self):
        """Setup placeholder encoder for testing"""
        logger.warning("Using placeholder text encoder (outputs random embeddings)")
        self._session = None
    
    def encode(
        self,
        text: str,
        return_attention_mask: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode text to embeddings.
        
        Args:
            text: Input text to encode
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Dictionary with 'embeddings' and optionally 'attention_mask'
        """
        if not self._loaded:
            self.load()
        
        # Tokenize
        tokens = self._tokenize(text)
        
        # Run encoder
        if self._session is not None:
            embeddings = self._run_onnx(tokens)
        else:
            embeddings = self._run_placeholder(tokens)
        
        result = {'embeddings': embeddings}
        
        if return_attention_mask:
            result['attention_mask'] = tokens.get('attention_mask', 
                                                   np.ones((1, embeddings.shape[1]), dtype=np.int64))
        
        return result
    
    def _tokenize(self, text: str) -> Dict[str, np.ndarray]:
        """Tokenize input text"""
        if hasattr(self._tokenizer, 'encode_plus'):
            # HuggingFace tokenizer
            encoded = self._tokenizer(
                text,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='np'
            )
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
            }
        else:
            # Basic tokenizer fallback
            return self._tokenizer.encode(text)
    
    def _run_onnx(self, tokens: Dict[str, np.ndarray]) -> np.ndarray:
        """Run ONNX inference"""
        input_names = [inp.name for inp in self._session.get_inputs()]
        
        # Prepare inputs
        onnx_inputs = {}
        for name in input_names:
            if 'input_ids' in name.lower():
                onnx_inputs[name] = tokens['input_ids'].astype(np.int64)
            elif 'attention' in name.lower() or 'mask' in name.lower():
                onnx_inputs[name] = tokens['attention_mask'].astype(np.int64)
        
        # Run
        outputs = self._session.run(None, onnx_inputs)
        
        return outputs[0]  # Return main embedding output
    
    def _run_placeholder(self, tokens: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate placeholder embeddings for testing"""
        batch_size = tokens['input_ids'].shape[0]
        seq_length = min(tokens['input_ids'].shape[1], self.max_length)
        
        # Generate deterministic pseudo-random embeddings based on input
        seed = int(np.sum(tokens['input_ids']) % 2**31)
        rng = np.random.RandomState(seed)
        
        # T5-XXL hidden size is 4096
        hidden_size = 4096
        
        embeddings = rng.randn(batch_size, seq_length, hidden_size).astype(np.float32)
        
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
        
        return embeddings
    
    def encode_batch(
        self,
        texts: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Encode multiple texts.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Batched embeddings
        """
        if not self._loaded:
            self.load()
        
        all_embeddings = []
        all_masks = []
        
        for text in texts:
            result = self.encode(text)
            all_embeddings.append(result['embeddings'])
            if 'attention_mask' in result:
                all_masks.append(result['attention_mask'])
        
        return {
            'embeddings': np.concatenate(all_embeddings, axis=0),
            'attention_mask': np.concatenate(all_masks, axis=0) if all_masks else None
        }
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension"""
        return 4096  # T5-XXL
    
    def unload(self):
        """Unload model to free memory"""
        self._session = None
        self._loaded = False
        logger.info("Unloaded text encoder")


class BasicTokenizer:
    """
    Basic tokenizer fallback when transformers is not available.
    """
    
    def __init__(self, max_length: int = 256):
        self.max_length = max_length
        self.vocab = {}  # Simple vocab will be built on the fly
        self._next_id = 1
    
    def encode(self, text: str) -> Dict[str, np.ndarray]:
        """Basic word-level tokenization"""
        words = text.lower().split()
        
        # Assign IDs
        input_ids = []
        for word in words[:self.max_length]:
            if word not in self.vocab:
                self.vocab[word] = self._next_id
                self._next_id += 1
            input_ids.append(self.vocab[word])
        
        # Pad
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_length:
            input_ids.append(0)
            attention_mask.append(0)
        
        return {
            'input_ids': np.array([input_ids], dtype=np.int64),
            'attention_mask': np.array([attention_mask], dtype=np.int64)
        }
