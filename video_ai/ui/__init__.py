"""
Video AI UI Module
Web-based and desktop interfaces for video generation.
"""

from .web_ui import WebUI, launch_ui
from .log_handler import UILogHandler

# Import aggressive generator components
try:
    from .aggressive_generator_tab import create_aggressive_generator_tab
    __all__ = ['WebUI', 'launch_ui', 'UILogHandler', 'create_aggressive_generator_tab']
except ImportError:
    __all__ = ['WebUI', 'launch_ui', 'UILogHandler']
