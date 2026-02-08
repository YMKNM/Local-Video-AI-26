#!/usr/bin/env python3
"""
Video AI UI Launcher

Launch the web-based user interface for video generation.

Usage:
    python run_ui.py
    python run_ui.py --port 8080
    python run_ui.py --share
"""

import argparse
import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_ai.ui import launch_ui


def main():
    parser = argparse.ArgumentParser(
        description="Video AI - Web UI Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_ui.py                    # Launch on default port 7860
    python run_ui.py --port 8080        # Launch on port 8080
    python run_ui.py --share            # Create public link
    python run_ui.py --debug            # Enable debug logging
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=7860,
        help='Port to run the web UI on (default: 7860)'
    )
    
    parser.add_argument(
        '--share', '-s',
        action='store_true',
        help='Create a public shareable link'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./outputs',
        help='Directory to save generated videos (default: ./outputs)'
    )
    
    parser.add_argument(
        '--config-dir', '-c',
        type=str,
        default=None,
        help='Path to config directory'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Print banner (ASCII-only to avoid cp1252 encoding errors on Windows cmd)
    print("")
    print("  ====================================================")
    print("    Video AI - Local AI Video Generation")
    print("    Web UI for GPU Acceleration")
    print("  ====================================================")
    print("")
    
    print(f"Starting web UI on port {args.port}...")
    print(f"Output directory: {args.output_dir}")
    if args.share:
        print("Public sharing enabled - a public URL will be generated")
    print()
    
    try:
        launch_ui(
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            share=args.share,
            port=args.port
        )
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nMissing dependencies. Please install:")
        print("    pip install gradio")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error starting UI: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
