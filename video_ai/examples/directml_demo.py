"""
Example: DirectML GPU Acceleration

This example demonstrates DirectML-specific features for AMD GPU
acceleration on Windows.
"""

import logging
from pathlib import Path

from video_ai.runtime import DirectMLSession, ONNXModelLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_directml_availability():
    """Check if DirectML is available and get GPU info"""
    print("\n" + "=" * 60)
    print("DirectML Availability Check")
    print("=" * 60)
    
    session = DirectMLSession()
    
    print(f"\nDirectML Available: {session.is_directml_available()}")
    
    # Get device info
    device_info = session.get_device_info()
    
    print(f"\nDevice Information:")
    print(f"  Name: {device_info.get('name', 'Unknown')}")
    print(f"  Device ID: {device_info.get('device_id', 'N/A')}")
    print(f"  Provider: {device_info.get('provider', 'N/A')}")
    print(f"  VRAM Total: {device_info.get('vram_total_gb', 'N/A'):.1f} GB")
    print(f"  VRAM Free: {device_info.get('vram_free_gb', 'N/A'):.1f} GB")
    
    # Get available providers
    providers = session.get_available_providers()
    print(f"\nAvailable Execution Providers:")
    for provider in providers:
        marker = "  ✓" if provider == device_info.get('provider') else "  "
        print(f"  {marker} {provider}")


def demonstrate_onnx_loading():
    """Demonstrate ONNX model loading with DirectML"""
    print("\n" + "=" * 60)
    print("ONNX Model Loading Demonstration")
    print("=" * 60)
    
    loader = ONNXModelLoader()
    
    # Get supported models
    supported = loader.get_supported_models()
    print(f"\nSupported Model Types:")
    for model_type in supported:
        print(f"  - {model_type}")
    
    # Check model paths
    models_dir = Path("./models")
    if models_dir.exists():
        print(f"\nAvailable ONNX Models in {models_dir}:")
        for onnx_file in models_dir.glob("**/*.onnx"):
            size_mb = onnx_file.stat().st_size / (1024 * 1024)
            print(f"  - {onnx_file.name} ({size_mb:.1f} MB)")
    else:
        print(f"\nModels directory not found. Run download_models.py first.")


def demonstrate_session_management():
    """Show DirectML session management"""
    print("\n" + "=" * 60)
    print("Session Management Demonstration")
    print("=" * 60)
    
    session = DirectMLSession()
    
    # Configure session options
    print("\nSession Configuration Options:")
    
    # Graph optimization levels
    optimization_levels = [
        ("None", "No optimizations"),
        ("Basic", "Basic optimizations (constant folding)"),
        ("Extended", "Extended optimizations (subgraph rewriting)"),
        ("All", "All optimizations (full optimization)")
    ]
    
    for level, description in optimization_levels:
        print(f"  - {level}: {description}")
    
    # Memory options
    print("\nMemory Configuration:")
    print("  - enable_memory_arena: Reuse memory allocations")
    print("  - enable_cpu_mem_arena: CPU memory pooling")
    print("  - memory_pattern: Optimize memory patterns")
    
    # Inference options
    print("\nInference Optimization:")
    print("  - parallel_execution: Use multiple threads")
    print("  - inter_op_threads: Threads between operations")
    print("  - intra_op_threads: Threads within operations")


def benchmark_directml():
    """Simple benchmark of DirectML performance"""
    print("\n" + "=" * 60)
    print("DirectML Performance Benchmark")
    print("=" * 60)
    
    import numpy as np
    import time
    
    session = DirectMLSession()
    
    if not session.is_directml_available():
        print("\nDirectML not available, skipping benchmark.")
        return
    
    # Create a test inference session
    try:
        import onnxruntime as ort
        
        # Create a simple test graph
        print("\nCreating test model...")
        
        # Note: In real usage, you would load an actual ONNX model
        # This is just a placeholder for demonstration
        
        print("\nBenchmark would test:")
        print("  1. Model loading time")
        print("  2. First inference (warmup)")
        print("  3. Average inference time over 10 runs")
        print("  4. Memory usage during inference")
        print("  5. GPU utilization")
        
        print("\nTo run the actual benchmark:")
        print("  1. Download models using: python download_models.py --all")
        print("  2. Run: python -m video_ai.examples.benchmark")
        
    except ImportError:
        print("\nONNX Runtime not installed. Install with:")
        print("  pip install onnxruntime-directml")


def main():
    """Run all DirectML demonstrations"""
    try:
        check_directml_availability()
        demonstrate_onnx_loading()
        demonstrate_session_management()
        benchmark_directml()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
