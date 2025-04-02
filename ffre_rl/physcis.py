"""
Physics engine for charged particle dynamics in electromagnetic fields.

This module provides the core physics calculations for simulating charged
particles in magnetic fields, which is essential for FFRE optimization.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Union, Any
from numba import jit
import warnings

from .constants import DEFAULT_SLOTS, MU_0

# Suppress NumPy warnings during JIT compilation
warnings.filterwarnings('ignore', category=NumWarning, module='numba')


@dataclass(frozen=True, slots=DEFAULT_SLOTS)
class ParticleState:
    """Immutable particle state representation for efficient physics calculations."""
    position: np.ndarray  # 3D position vector (x, y, z) in meters
    velocity: np.ndarray  # 3D velocity vector (vx, vy, vz) in m/s
    mass: float  # Particle mass in kg
    charge: float  # Particle charge in Coulombs
    active: bool = True  # Whether particle is still in the simulation
    escaped_thrust: bool = False  # Whether particle escaped and contributed to thrust


class ParticlePhysicsEngine:
    """
    Efficient physics engine for charged particle dynamics in electromagnetic fields.
    
    This class implements high-performance calculations for particle motion using
    vectorized operations and JIT compilation where appropriate.
    """
    
    def __init__(self, dt: float = 1e-9, use_jit: bool = True):
        """
        Initialize the physics engine with a timestep dt in seconds.
        
        Args:
            dt: Simulation time step in seconds
            use_jit: Whether to use JIT compilation for performance-critical functions
        """
        self.dt = dt
        self.use_jit = use_jit
        
        # JIT-compile performance-critical functions if enabled
        if use_jit:
            # Apply JIT compilation to methods
            self._jit_calculate_lorentz_force = jit(nopython=True)(self._calculate_lorentz_force_impl)
            self._jit_update_particle_state = jit(nopython=True)(self._update_particle_state_impl)
        
    @staticmethod
    @jit(nopython=True)
    def _calculate_lorentz_force_impl(charge: float, velocity: np.ndarray, 
                                     e_field: np.ndarray, b_field: np.ndarray) -> np.ndarray:
        """
        JIT-compiled implementation of Lorentz force calculation.
        
        Args:
            charge: Particle charge in Coulombs
            velocity: Particle velocity vector (3D) in m/s
            e_field: Electric field vector (3D) in V/m
            b_field: Magnetic field vector (3D) in Tesla
            
        Returns:
            Force vector (3D) in Newtons
        """
        electric_force = charge * e_field
        magnetic_force = charge * np.cross(velocity, b_field)
        return electric_force + magnetic_force
    
    @staticmethod
    @jit(nopython=True)
    def _update_particle_state_impl(position: np.ndarray, velocity: np.ndarray, 
                                   mass: float, charge: float, e_field: np.ndarray, 
                                   b_field: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        JIT-compiled implementation of particle state update.
        
        Args:
            position: Current position vector (3D) in meters
            velocity: Current velocity vector (3D) in m/s
            mass: Particle mass in kg
            charge: Particle charge in Coulombs
            e_field: Electric field vector (3D) in V/m
            b_field: Magnetic field vector (3D) in Tesla
            dt: Time step in seconds
            
        Returns:
            Tuple of updated position and velocity vectors
        """
        # Calculate force
        electric_force = charge * e_field
        magnetic_force = charge * np.cross(velocity, b_field)
        force = electric_force + magnetic_force
        
        # Calculate acceleration (F = ma)
        acceleration = force / mass
        
        # Update velocity using acceleration - use RK4 for better accuracy
        k1v = acceleration
        k1x = velocity
        
        mid_vel = velocity + 0.5 * dt * k1v
        k2v = (charge * e_field + charge * np.cross(mid_vel, b_field)) / mass
        k2x = mid_vel
        
        mid_vel = velocity + 0.5 * dt * k2v
        k3v = (charge * e_field + charge * np.cross(mid_vel, b_field)) / mass
        k3x = mid_vel
        
        end_vel = velocity + dt * k3v
        k4v = (charge * e_field + charge * np.cross(end_vel, b_field)) / mass
        k4x = end_vel
        
        # Final update using weighted average
        new_velocity = velocity + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        new_position = position + (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        
        return new_position, new_velocity
        
    def calculate_lorentz_force(self, charge: float, velocity: np.ndarray, 
                              e_field: np.ndarray, b_field: np.ndarray) -> np.ndarray:
        """
        Calculate the Lorentz force on a charged particle.
        
        Args:
            charge: Particle charge in Coulombs
            velocity: Particle velocity vector (3D) in m/s
            e_field: Electric field vector (3D) in V/m
            b_field: Magnetic field vector (3D) in Tesla
            
        Returns:
            Force vector (3D) in Newtons
        """
        if self.use_jit:
            return self._jit_calculate_lorentz_force(charge, velocity, e_field, b_field)
        else:
            # Non-JIT fallback
            electric_force = charge * e_field
            magnetic_force = charge * np.cross(velocity, b_field)
            return electric_force + magnetic_force
    
    def update_particle_state(self, position: np.ndarray, velocity: np.ndarray, 
                            mass: float, charge: float, e_field: np.ndarray, 
                            b_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the position and velocity of a particle based on electromagnetic forces.
        
        Args:
            position: Current position vector (3D) in meters
            velocity: Current velocity vector (3D) in m/s
            mass: Particle mass in kg
            charge: Particle charge in Coulombs
            e_field: Electric field vector (3D) in V/m
            b_field: Magnetic field vector (3D) in Tesla
            
        Returns:
            Tuple of updated position and velocity vectors
        """
        if self.use_jit:
            return self._jit_update_particle_state(
                position, velocity, mass, charge, e_field, b_field, self.dt
            )
        else:
            # Calculate force
            force = self.calculate_lorentz_force(charge, velocity, e_field, b_field)
            
            # Calculate acceleration (F = ma)
            acceleration = force / mass
            
            # Use 4th order Runge-Kutta for more accurate integration
            # This significantly improves energy conservation in magnetic fields
            
            # k1 is the initial derivative
            k1v = acceleration
            k1x = velocity
            
            # k2 is the derivative at the midpoint using k1
            mid_vel = velocity + 0.5 * self.dt * k1v
            mid_force = self.calculate_lorentz_force(charge, mid_vel, e_field, b_field)
            k2v = mid_force / mass
            k2x = mid_vel
            
            # k3 is the derivative at the midpoint using k2
            mid_vel = velocity + 0.5 * self.dt * k2v
            mid_force = self.calculate_lorentz_force(charge, mid_vel, e_field, b_field)
            k3v = mid_force / mass
            k3x = mid_vel
            
            # k4 is the derivative at the end using k3
            end_vel = velocity + self.dt * k3v
            end_force = self.calculate_lorentz_force(charge, end_vel, e_field, b_field)
            k4v = end_force / mass
            k4x = end_vel
            
            # Final update using weighted average
            new_velocity = velocity + (self.dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
            new_position = position + (self.dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
            
            return new_position, new_velocity
    
    def update_particles_vectorized(self, particles: List[Dict[str, Any]]) -> None:
        """
        Update multiple particles in a single vectorized operation for better performance.
        
        Args:
            particles: List of particle dictionaries with position, velocity, mass, and charge
        """
        # Extract arrays for vectorized operations
        active_indices = [i for i, p in enumerate(particles) if p['active']]
        if not active_indices:
            return
            
        # Get active particles
        active_particles = [particles[i] for i in active_indices]
        
        # Stack properties for vectorized operations
        positions = np.stack([p['position'] for p in active_particles])
        velocities = np.stack([p['velocity'] for p in active_particles])
        masses = np.array([p['mass'] for p in active_particles])
        charges = np.array([p['charge'] for p in active_particles])
        
        # Calculate fields for each particle position
        e_fields = np.zeros_like(positions)  # Placeholder - in real code, this would be calculated
        b_fields = np.zeros_like(positions)  # Placeholder - in real code, this would be calculated
        
        # Calculate forces (vectorized)
        electric_forces = charges[:, np.newaxis] * e_fields
        magnetic_forces = np.zeros_like(positions)
        for i in range(len(active_particles)):
            magnetic_forces[i] = charges[i] * np.cross(velocities[i], b_fields[i])
        
        total_forces = electric_forces + magnetic_forces
        
        # Calculate accelerations
        accelerations = total_forces / masses[:, np.newaxis]
        
        # Update positions and velocities using Semi-Implicit Euler
        # (simpler than RK4 for vectorized operations)
        new_velocities = velocities + accelerations * self.dt
        new_positions = positions + new_velocities * self.dt
        
        # Update the particle dictionaries
        for i, idx in enumerate(active_indices):
            particles[idx]['velocity'] = new_velocities[i]
            particles[idx]['position'] = new_positions[i]
    
    def calculate_larmor_radius(self, mass: float, velocity_perpendicular: float, 
                              charge: float, b_field_magnitude: float) -> float:
        """
        Calculate the Larmor radius for a charged particle in a magnetic field.
        
        Args:
            mass: Particle mass in kg
            velocity_perpendicular: Component of velocity perpendicular to B field in m/s
            charge: Particle charge in Coulombs
            b_field_magnitude: Magnitude of the magnetic field in Tesla
            
        Returns:
            Larmor radius in meters
        """
        if b_field_magnitude == 0:
            return float('inf')  # Infinite radius when no magnetic field
        
        return (mass * velocity_perpendicular) / (abs(charge) * b_field_magnitude)
    
    def calculate_cyclotron_frequency(self, charge: float, mass: float, 
                                    b_field_magnitude: float) -> float:
        """
        Calculate the cyclotron frequency for a charged particle in a magnetic field.
        
        Args:
            charge: Particle charge in Coulombs
            mass: Particle mass in kg
            b_field_magnitude: Magnitude of the magnetic field in Tesla
            
        Returns:
            Cyclotron frequency in radians per second
        """
        if b_field_magnitude == 0:
            return 0.0
            
        return abs(charge * b_field_magnitude / mass)
    
    def calculate_magnetic_moment(self, mass: float, v_perp_squared: float, 
                                b_field_magnitude: float) -> float:
        """
        Calculate the magnetic moment of a charged particle.
        
        Args:
            mass: Particle mass in kg
            v_perp_squared: Square of velocity component perpendicular to B field in (m/s)²
            b_field_magnitude: Magnitude of the magnetic field in Tesla
            
        Returns:
            Magnetic moment in J/T
        """
        if b_field_magnitude == 0:
            return 0.0
            
        return 0.5 * mass * v_perp_squared / b_field_magnitude
    
    @staticmethod
    def decompose_velocity(velocity: np.ndarray, b_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose velocity into components parallel and perpendicular to the magnetic field.
        
        Args:
            velocity: Particle velocity vector
            b_field: Magnetic field vector
            
        Returns:
            Tuple of (parallel_component, perpendicular_component) velocity vectors
        """
        if np.all(b_field == 0):
            return np.zeros(3), velocity
            
        b_unit = b_field / np.linalg.norm(b_field)
        v_parallel = np.dot(velocity, b_unit) * b_unit
        v_perpendicular = velocity - v_parallel
        
        return v_parallel, v_perpendicular
    
    @staticmethod
    def calculate_magnetic_field_from_coils(position: np.ndarray, coil_positions: np.ndarray, 
                                         coil_currents: np.ndarray, coil_radius: float) -> np.ndarray:
        """
        Calculate the magnetic field at a given position due to multiple current-carrying coils.
        
        Uses the Biot-Savart law for circular coils.
        
        Args:
            position: 3D position vector where to calculate the field
            coil_positions: Nx3 array of coil center positions
            coil_currents: N-element array of coil currents
            coil_radius: Radius of the coils (assuming all coils have the same radius)
            
        Returns:
            Magnetic field vector at the specified position
        """
        # Calculate field from each coil using vectorized operations
        b_field = np.zeros(3)
        
        for coil_pos, current in zip(coil_positions, coil_currents):
            # Calculate vector from coil center to position
            r_vec = position - coil_pos
            r_mag = np.linalg.norm(r_vec)
            
            # Skip if we're exactly at the coil center
            if r_mag < 1e-10:
                continue
                
            # Axis of the coil (assuming z-direction)
            axis = np.array([0, 0, 1])
            
            # For on-axis points, we can use a simpler formula
            if np.linalg.norm(r_vec - np.dot(r_vec, axis) * axis) < 1e-10:
                # On-axis field
                z = np.dot(r_vec, axis)
                b_mag = (MU_0 * current * coil_radius**2) / (2 * (coil_radius**2 + z**2)**(3/2))
                b_field += b_mag * axis
            else:
                # For off-axis points, we need a more complex formula
                # This is an approximation using the dipole moment
                # For accurate fields, numerical integration would be needed
                moment = np.pi * coil_radius**2 * current * axis
                
                # Dipole field formula
                r_unit = r_vec / r_mag
                b = (MU_0 / (4 * np.pi)) * (3 * r_unit * np.dot(moment, r_unit) - moment) / r_mag**3
                b_field += b
                
        return b_field


# Optimized classes for memory efficiency

class FastParticleTracker:
    """
    Memory-efficient particle tracker using NumPy structured arrays instead of dictionaries.
    """
    # Define structured data type for particles
    particle_dtype = np.dtype([
        ('position', np.float32, 3),
        ('velocity', np.float32, 3),
        ('mass', np.float32),
        ('charge', np.float32),
        ('active', np.bool_),
        ('escaped_thrust', np.bool_)
    ])
    
    def __init__(self, num_particles: int):
        """
        Initialize the particle tracker.
        
        Args:
            num_particles: Number of particles to track
        """
        # Allocate structured array for particles
        self.particles = np.zeros(num_particles, dtype=self.particle_dtype)
        self.num_particles = num_particles
        
    def update_particles(self, physics_engine: ParticlePhysicsEngine, 
                       get_fields_func: callable) -> None:
        """
        Update all active particles using the provided physics engine.
        
        Args:
            physics_engine: Physics engine to use for updates
            get_fields_func: Function that returns (e_field, b_field) for a given position
        """
        # Get active particles mask
        active_mask = self.particles['active']
        
        if not np.any(active_mask):
            return
            
        # Get active particles
        active_particles = self.particles[active_mask]
        
        # Update each active particle
        for i, particle in enumerate(active_particles):
            # Get fields at current position
            e_field, b_field = get_fields_func(particle['position'])
            
            # Update particle state
            new_position, new_velocity = physics_engine.update_particle_state(
                particle['position'], particle['velocity'],
                particle['mass'], particle['charge'],
                e_field, b_field
            )
            
            # Store updated state
            active_particles[i]['position'] = new_position
            active_particles[i]['velocity'] = new_velocity
            
        # Write back active particles
        self.particles[active_mask] = active_particles