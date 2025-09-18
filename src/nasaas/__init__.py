"""
NASaaS: Neural Architecture Search as a Service
Core package for automated neural architecture search and deployment.
"""

__version__ = "0.1.0"
__author__ = "FailedToAchieveOrbit"
__email__ = "sawadall@asu.edu"

# Core imports - these will be implemented in subsequent commits
try:
    from .client import NASClient
    from .core.nas_engine import NASEngine
    from .core.search_algorithms import DARTSSearcher, ENASSearcher, ProgressiveNASSearcher
    from .utils.task_parser import TaskParser
    from .deployment.mcp_manager import MCPDeploymentManager
except ImportError:
    # During initial development, modules may not exist yet
    pass

__all__ = [
    "NASClient",
    "NASEngine", 
    "DARTSSearcher",
    "ENASSearcher",
    "ProgressiveNASSearcher",
    "TaskParser",
    "MCPDeploymentManager"
]

# Package metadata
__title__ = "nasaas"
__description__ = "Neural Architecture Search as a Service - Autonomous neural network design from natural language"
__url__ = "https://github.com/FailedToAchieveOrbit/nasaas"
__license__ = "MIT"
__copyright__ = "Copyright 2025 FailedToAchieveOrbit"