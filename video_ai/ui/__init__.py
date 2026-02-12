"""
Video AI UI Module
Web-based and desktop interfaces for video generation.
"""

from .web_ui import WebUI, launch_ui
from .log_handler import UILogHandler

# Import aggressive generator components
try:
    from .aggressive_generator_tab import create_aggressive_generator_tab
except ImportError:
    create_aggressive_generator_tab = None

# Import DeepSeek tab
try:
    from .deepseek_tab import create_deepseek_tab
except ImportError:
    create_deepseek_tab = None

__all__ = ['WebUI', 'launch_ui', 'UILogHandler', 'create_aggressive_generator_tab', 'create_deepseek_tab']
