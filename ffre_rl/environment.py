"""
Reinforcement learning environment for FFRE magnetic field optimization.

This module provides a Gymnasium-compatible environment that simulates
alpha particles in magnetic fields for FFRE optimization.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from collections import deque
from dataclasses import dataclass, field
import warnings

# Local imports
from .physics import ParticlePhysicsEngine, FastParticleTracker
from .constants import (
    ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, AM241_ENERGY, DEFAULT_SLOTS
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass(slots=DEFAULT_SLOTS)
class ParticleConfig:
    """Configuration for particle generation."""
    mass: float
    charge: float
    energy: float
    num_particles: int
    emission_radius: float = 0.0  # For point source
    direction_bias: float = 0.0   # 0.0 = isotropic, 1.0 = fully forward-directed
    
    @property
    def velocity_magnitude(self) -> float:
        """Calculate velocity magnitude from energy."""
        return np.sqrt(2 * self.energy / self.mass)


@dataclass(slots=DEFAULT_SLOTS)
class FFREConfig:
    """Configuration for FFRE environment."""
    chamber_length: float = 0.5      # meters
    chamber_radius: float = 0.15     # meters
    max_field_strength: float = 5.0  # Tesla
    n_field_coils: int = 5
    max_steps: int = 200
    dt: float = 1e-9                 # seconds
    
    # Particle configuration
    particle_config: ParticleConfig = field(default_factory=lambda: ParticleConfig(
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        energy=AM241_ENERGY,
        num_particles=100
    ))
    
    # Reward configuration
    thrust_weight: float = 5.0
    efficiency_weight: float = 3.0
    escape_weight: float = 2.0
    wall_collision_penalty: float = 0.1
    
    # Physics optimization
    use_jit: bool = True
    use_vectorized_updates: bool = True
    use_structured_arrays: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.chamber_length > 0, "Chamber length must be positive"
        assert self.chamber_radius > 0, "Chamber radius must be positive"
        assert self.max_field_strength > 0, "Maximum field strength must be positive"
        assert self.n_field_coils > 0, "Number of field coils must be positive"
        assert self.max_steps > 0, "Maximum steps must be positive"
        assert self.dt > 0, "Time step must be positive"
        assert self.particle_config.num_particles > 0, "Number of particles must be positive"


class FFREEnvironment(gym.Env):
    """
    Reinforcement Learning environment for optimizing magnetic field configurations
    in a Fission Fragment Rocket Engine.
    
    This environment simulates the behavior of charged particles (alpha particles or
    fission fragments) in a cylindrical vacuum chamber with controllable magnetic
    field coils. The goal is to configure the magnetic fields to maximize thrust
    while efficiently guiding particles out of the chamber.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, config: Optional[FFREConfig] = None):
        """
        Initialize the FFRE environment.
        
        Args:
            config: Configuration object for the environment
        """
        super().__init__()
        
        # Use default config if none provided
        self.config = config or FFREConfig()
        self.config.validate()
        
        # Initialize state variables
        self.steps = 0
        self.thrust = 0.0
        self.efficiency = 0.0
        self.escape_rate = 0.0
        
        # Create physics engine
        self.physics_engine = ParticlePhysicsEngine(
            dt=self.config.dt, 
            use_jit=self.config.use_jit
        )
        
        # Initialize particles
        if self.config.use_structured_arrays:
            self.particle_tracker = FastParticleTracker(self.config.particle_config.num_particles)
            self._init_structured_particles()
            self.particles = None  # Not used in structured mode
        else:
            self.particles = []
            self.particle_tracker = None
        
        # Magnetic field configuration
        self.magnetic_field_config = np.zeros(self.config.n_field_coils)
        
        # Field coil positions (normalized along chamber length)
        self.field_coil_positions = np.linspace(0, 1, self.config.n_field_coils)
        
        # Action space: Control the magnetic field at each coil location
        # Each coil can have a field strength from 0 to max_field_strength
        self.action_space = spaces.Box(
            low=0, 
            high=1,  # We'll scale this to max_field_strength
            shape=(self.config.n_field_coils,),
            dtype=np.float32
        )
        
        # Observation space: Current state representation
        # - Current magnetic field configuration
        # - Particle distribution statistics (e.g., average position, velocity)
        # - Thrust metrics
        observation_dim = (
            self.config.n_field_coils +  # Magnetic field strength at each coil
            6 +                          # Average particle position (3) and velocity (3)
            3                            # Thrust, efficiency, and escape rate metrics
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(observation_dim,),
            dtype=np.float32
        )
        
        # Initialize buffers for performance statistics
        self.thrust_history = deque(maxlen=10)
        self.efficiency_history = deque(maxlen=10)
        self.escape_rate_history = deque(maxlen=10)
        
        logger.info(f"FFRE Environment initialized with chamber dimensions: "
                   f"{self.config.chamber_length}m x {self.config.chamber_radius}m radius")
    
    def _init_structured_particles(self) -> None:
        """Initialize particles using structured arrays for memory efficiency."""
        p_config = self.config.particle_config
        
        # Get velocity magnitude from energy
        velocity_magnitude = p_config.velocity_magnitude
        
        # Set initial particle states
        for i in range(p_config.num_particles):
            # Random position near the source (start of chamber)
            position = np.array([
                0.05 * self.config.chamber_length * np.random.random(),  # x
                (np.random.random() - 0.5) * p_config.emission_radius * self.config.chamber_radius,  # y
                (np.random.random() - 0.5) * p_config.emission_radius * self.config.chamber_radius   # z
            ], dtype=np.float32)
            
            # Random velocity direction biased toward chamber axis
            if p_config.direction_bias > 0:
                # Biased direction sampling for better efficiency
                # Generate using rejection sampling for uniform distribution on sphere
                # biased toward positive x-axis
                while True:
                    # Random direction on unit sphere
                    phi = 2 * np.pi * np.random.random()
                    theta = np.arccos(2 * np.random.random() -