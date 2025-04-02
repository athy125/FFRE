"""
FFRE-RL: Reinforcement Learning for Fission Fragment Rocket Engine Optimization

This package provides tools for optimizing magnetic field configurations
in Fission Fragment Rocket Engines using reinforcement learning.
"""

__version__ = "1.0.0"

# Import public API components
from .constants import (
    ELECTRON_CHARGE, ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS,
    AM241_ENERGY, THORIUM232_ENERGY
)

from .physics import ParticlePhysicsEngine
from .environment import FFREEnvironment
from .agent import PPOAgent
from .visualization import MagneticFieldVisualization
from .evaluation import TrajectoryRecorder
from .utils import run_training, evaluate_agent

# Set up package namespace
__all__ = [
    # Constants
    'ELECTRON_CHARGE', 'ALPHA_PARTICLE_CHARGE', 'ALPHA_PARTICLE_MASS',
    'AM241_ENERGY', 'THORIUM232_ENERGY',
    
    # Classes
    'ParticlePhysicsEngine', 'FFREEnvironment', 'PPOAgent',
    'MagneticFieldVisualization', 'TrajectoryRecorder',
    
    # Functions
    'run_training', 'evaluate_agent'
]