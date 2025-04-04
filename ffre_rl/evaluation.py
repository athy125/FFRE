"""
Evaluation utilities for FFRE optimization.

This module provides tools for evaluating and analyzing trained policies
for FFRE magnetic field optimization.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import time
import os
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)


class TrajectoryRecorder:
    """
    Records particle trajectories for visualization and analysis.
    
    This class provides methods for recording and analyzing particle trajectories
    during FFRE simulation, enabling detailed evaluation of policy performance.
    """
    
    def __init__(self, env, max_particles: Optional[int] = None, 
               record_fields: bool = True):
        """
        Initialize the trajectory recorder.
        
        Args:
            env: FFRE environment
            max_particles: Maximum number of particles to record (None for all)
            record_fields: Whether to record field values along trajectories
        """
        self.env = env
        self.max_particles = max_particles
        self.record_fields = record_fields
        self.trajectories = []
        self.field_data = []
        
        # Statistics
        self.statistics = {}
    
    def record_episode(self, agent, deterministic: bool = True) -> Dict[str, Any]:
        """
        Record particle trajectories for a full episode.
        
        Args:
            agent: RL agent
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary of episode information
        """
        # Reset environment and trajectories
        state, _ = self.env.reset()
        self.trajectories = []
        self.field_data = []
        
        # Get number of particles to record
        if self.max_particles is None:
            if hasattr(self.env, 'config') and hasattr(self.env.config, 'particle_config'):
                n_particles = self.env.config.particle_config.num_particles
            else:
                n_particles = len(self.env.get_all_particles())
        else:
            n_particles = min(self.max_particles, len(self.env.get_all_particles()))
        
        # Initialize trajectory tracking for each particle
        particle_trajectories = [[] for _ in range(n_particles)]
        
        # Initialize field data if recording
        if self.record_fields:
            field_data = [[] for _ in range(n_particles)]
            # Get field function from environment
            if hasattr(self.env, 'get_field_function'):
                field_function = self.env.get_field_function()
            else:
                # Default to zero field if not available
                field_function = lambda pos: np.zeros(3)
        
        # Record initial states
        particles = self.env.get_all_particles()
        for i in range(n_particles):
            if i < len(particles):
                # Record initial state
                particle_trajectories[i].append({
                    'position': particles[i]['position'].copy(),
                    'velocity': particles[i]['velocity'].copy(),
                    'mass': particles[i]['mass'],
                    'charge': particles[i]['charge'],
                    'active': particles[i]['active'],
                    'escaped_thrust': particles[i]['escaped_thrust'],
                    'step': 0
                })
                
                # Record field if enabled
                if self.record_fields:
                    field = field_function(particles[i]['position'])
                    field_data[i].append({
                        'position': particles[i]['position'].copy(),
                        'field': field.copy(),
                        'field_magnitude': np.linalg.norm(field),
                        'step': 0
                    })
        
        # Run episode
        episode_reward = 0
        done = False
        step = 0
        info = {}
        
        while not done and step < getattr(self.env, 'max_steps', 1000):
            # Get action from agent
            action, _ = agent.get_action(state, deterministic=deterministic)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Record particle states
            particles = self.env.get_all_particles()
            for i in range(n_particles):
                if i < len(particles) and particles[i]['active']:
                    # Record state
                    particle_trajectories[i].append({
                        'position': particles[i]['position'].copy(),
                        'velocity': particles[i]['velocity'].copy(),
                        'mass': particles[i]['mass'],
                        'charge': particles[i]['charge'],
                        'active': particles[i]['active'],
                        'escaped_thrust': particles[i]['escaped_thrust'],
                        'step': step + 1
                    })
                    
                    # Record field if enabled
                    if self.record_fields:
                        field = field_function(particles[i]['position'])
                        field_data[i].append({
                            'position': particles[i]['position'].copy(),
                            'field': field.copy(),
                            'field_magnitude': np.linalg.norm(field),
                            'step': step + 1
                        })
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step += 1
        
        # Store trajectories (only those with at least 2 points)
        self.trajectories = [traj for traj in particle_trajectories if len(traj) >= 2]
        
        # Store field data if recorded
        if self.record_fields:
            self.field_data = [data for data in field_data if len(data) >= 2]
        
        # Compute statistics
        self._compute_statistics(info, action)
        
        # Return episode info
        return {
            'reward': episode_reward,
            'steps': step,
            'thrust': info.get('thrust', 0),
            'efficiency': info.get('efficiency', 0),
            'escape_rate': info.get('escape_rate', 0),
            'wall_collisions': info.get('wall_collisions', 0),
            'active_particles': info.get('active_particles', 0),
            'final_field_config': action
        }
    
    def _compute_statistics(self, info: Dict[str, Any], final_field_config: np.ndarray) -> None:
        """
        Compute statistics from recorded trajectories.
        
        Args:
            info: Info dictionary from environment
            final_field_config: Final magnetic field configuration
        """
        # Basic statistics from info
        self.statistics['thrust'] = info.get('thrust', 0)
        self.statistics['efficiency'] = info.get('efficiency', 0)
        self.statistics['escape_rate'] = info.get('escape_rate', 0)
        self.statistics['wall_collisions'] = info.get('wall_collisions', 0)
        self.statistics['active_particles'] = info.get('active_particles', 0)
        self.statistics['final_field_config'] = final_field_config.copy()
        
 # Compute advanced statistics
        # Count outcomes
        n_escaped = sum(1 for traj in self.trajectories if traj[-1].get('escaped_thrust', False))
        n_wall_collisions = sum(1 for traj in self.trajectories if not traj[-1].get('active', False) and not traj[-1].get('escaped_thrust', False))
        n_active = sum(1 for traj in self.trajectories if traj[-1].get('active', False))
        
        # Store counts
        self.statistics['n_trajectories'] = len(self.trajectories)
        self.statistics['n_escaped'] = n_escaped
        self.statistics['n_wall_collisions'] = n_wall_collisions
        self.statistics['n_active'] = n_active
        
        # Calculate percentages
        if len(self.trajectories) > 0:
            self.statistics['escaped_percent'] = 100 * n_escaped / len(self.trajectories)
            self.statistics['wall_collision_percent'] = 100 * n_wall_collisions / len(self.trajectories)
            self.statistics['active_percent'] = 100 * n_active / len(self.trajectories)
        else:
            self.statistics['escaped_percent'] = 0
            self.statistics['wall_collision_percent'] = 0
            self.statistics['active_percent'] = 0
        
        # Calculate trajectory statistics
        if len(self.trajectories) > 0:
            # Calculate trajectory lengths
            trajectory_lengths = [len(traj) for traj in self.trajectories]
            self.statistics['avg_trajectory_length'] = np.mean(trajectory_lengths)
            self.statistics['min_trajectory_length'] = np.min(trajectory_lengths)
            self.statistics['max_trajectory_length'] = np.max(trajectory_lengths)
            
            # Calculate travel distances
            travel_distances = []
            for traj in self.trajectories:
                positions = np.array([state['position'] for state in traj])
                if len(positions) > 1:
                    # Calculate distance along path
                    diff = positions[1:] - positions[:-1]
                    distances = np.sqrt(np.sum(diff**2, axis=1))
                    total_distance = np.sum(distances)
                    travel_distances.append(total_distance)
            
            if travel_distances:
                self.statistics['avg_travel_distance'] = np.mean(travel_distances)
                self.statistics['min_travel_distance'] = np.min(travel_distances)
                self.statistics['max_travel_distance'] = np.max(travel_distances)
            
            # Calculate final velocities
            final_velocities = []
            for traj in self.trajectories:
                if len(traj) > 0:
                    final_vel = traj[-1]['velocity']
                    vel_mag = np.linalg.norm(final_vel)
                    final_velocities.append(vel_mag)
            
            if final_velocities:
                self.statistics['avg_final_velocity'] = np.mean(final_velocities)
                self.statistics['min_final_velocity'] = np.min(final_velocities)
                self.statistics['max_final_velocity'] = np.max(final_velocities)
        
        # Field statistics if recorded
        if self.record_fields and len(self.field_data) > 0:
            # Calculate field statistics
            field_magnitudes = []
            for data in self.field_data:
                for point in data:
                    field_magnitudes.append(point['field_magnitude'])
            
            if field_magnitudes:
                self.statistics['avg_field_magnitude'] = np.mean(field_magnitudes)
                self.statistics['min_field_magnitude'] = np.min(field_magnitudes)
                self.statistics['max_field_magnitude'] = np.max(field_magnitudes)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get computed statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.statistics
    
    def get_trajectories(self) -> List[List[Dict[str, Any]]]:
        """
        Get recorded trajectories.
        
        Returns:
            List of particle trajectories
        """
        return self.trajectories
    
    def get_field_data(self) -> List[List[Dict[str, Any]]]:
        """
        Get recorded field data.
        
        Returns:
            List of field data
        """
        return self.field_data
    
    def save_trajectories(self, filepath: str) -> None:
        """
        Save trajectories to file.
        
        Args:
            filepath: Path to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save trajectories
        np.savez_compressed(filepath, 
                          trajectories=self.trajectories, 
                          field_data=self.field_data if self.record_fields else [],
                          statistics=self.statistics)
        
        logger.info(f"Trajectories saved to {filepath}")
    
    @staticmethod
    def load_trajectories(filepath: str) -> 'TrajectoryRecorder':
        """
        Load trajectories from file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            TrajectoryRecorder instance
        """
        # Load data
        data = np.load(filepath, allow_pickle=True)
        
        # Create recorder
        recorder = TrajectoryRecorder(None)
        
        # Set data
        recorder.trajectories = data['trajectories'].tolist()
        recorder.field_data = data['field_data'].tolist() if 'field_data' in data else []
        recorder.statistics = data['statistics'].item() if 'statistics' in data else {}
        
        logger.info(f"Trajectories loaded from {filepath}")
        
        return recorder


def evaluate_agent(agent, env, num_episodes: int = 10, deterministic: bool = True,
                 record_trajectories: bool = True, max_particles: Optional[int] = None,
                 render: bool = False) -> Tuple[Dict[str, List[float]], TrajectoryRecorder, np.ndarray]:
    """
    Evaluate a trained agent.
    
    Args:
        agent: Trained RL agent
        env: FFRE environment
        num_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        record_trajectories: Whether to record trajectories
        max_particles: Maximum number of particles to record
        render: Whether to render environment
        
    Returns:
        Tuple of (metrics, trajectory_recorder, best_field_config)
    """
    # Setup trajectory recorder if requested
    recorder = TrajectoryRecorder(env, max_particles=max_particles) if record_trajectories else None
    
    # Metrics to track
    metrics = {
        'rewards': [],
        'thrust_values': [],
        'efficiency_values': [],
        'escape_rates': [],
        'wall_collision_rates': [],
        'field_configs': []
    }
    
    # Best configuration seen so far
    best_reward = -np.inf
    best_field_config = None
    best_info = None
    
    for episode in range(num_episodes):
        # Record episode with deterministic actions
        if record_trajectories:
            episode_info = recorder.record_episode(agent, deterministic=deterministic)
        else:
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_info = {}
            
            # Run episode
            done = False
            step = 0
            
            while not done and step < getattr(env, 'max_steps', 1000):
                # Get action from agent
                action, _ = agent.get_action(state, deterministic=deterministic)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                step += 1
                
                # Render if enabled
                if render:
                    env.render()
            
            # Store episode info
            episode_info = {
                'reward': episode_reward,
                'steps': step,
                'thrust': info.get('thrust', 0),
                'efficiency': info.get('efficiency', 0),
                'escape_rate': info.get('escape_rate', 0),
                'wall_collisions': info.get('wall_collisions', 0),
                'active_particles': info.get('active_particles', 0),
                'final_field_config': action
            }
        
        # Store metrics
        metrics['rewards'].append(episode_info['reward'])
        metrics['thrust_values'].append(episode_info['thrust'])
        metrics['efficiency_values'].append(episode_info['efficiency'])
        metrics['escape_rates'].append(episode_info['escape_rate'])
        metrics['wall_collision_rates'].append(
            episode_info.get('wall_collisions', 0) / getattr(env, 'num_particles', 1)
        )
        metrics['field_configs'].append(episode_info['final_field_config'])
        
        # Check if best configuration
        if episode_info['reward'] > best_reward:
            best_reward = episode_info['reward']
            best_field_config = episode_info['final_field_config'].copy()
            best_info = {k: v for k, v in episode_info.items() if k != 'final_field_config'}
        
        # Log episode results
        logger.info(f"Evaluation Episode {episode}: "
                   f"Reward={episode_info['reward']:.3f}, "
                   f"Thrust={episode_info['thrust']:.3e}, "
                   f"Efficiency={episode_info['efficiency']:.3e}, "
                   f"Escape Rate={episode_info['escape_rate']:.3f}")
    
    # Log best configuration
    logger.info(f"Best configuration: {best_field_config}")
    logger.info(f"Best reward: {best_reward:.3f}")
    logger.info(f"Best thrust: {best_info.get('thrust', 0):.3e}")
    logger.info(f"Best efficiency: {best_info.get('efficiency', 0):.3e}")
    logger.info(f"Best escape rate: {best_info.get('escape_rate', 0):.3f}")
    
    return metrics, recorder, best_field_config


class PerformanceProfiler:
    """
    Profiler for measuring performance of FFRE simulation and RL algorithms.
    
    This class provides tools for measuring and analyzing the performance of
    FFRE simulations and RL algorithms, helping identify bottlenecks and optimize
    the code for efficiency.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance profiler.
        
        Args:
            window_size: Size of the moving average window
        """
        self.window_size = window_size
        self.reset()
    
    def reset(self) -> None:
        """Reset profiler."""
        self.step_times = deque(maxlen=self.window_size)
        self.inference_times = deque(maxlen=self.window_size)
        self.physics_times = deque(maxlen=self.window_size)
        self.update_times = deque(maxlen=self.window_size)
        self.total_steps = 0
        self.total_updates = 0
        self.start_time = time.time()
    
    def record_step(self, step_time: float, inference_time: float, physics_time: float) -> None:
        """
        Record timing for a single step.
        
        Args:
            step_time: Total time for step
            inference_time: Time for policy inference
            physics_time: Time for physics simulation
        """
        self.step_times.append(step_time)
        self.inference_times.append(inference_time)
        self.physics_times.append(physics_time)
        self.total_steps += 1
    
    def record_update(self, update_time: float) -> None:
        """
        Record timing for policy update.
        
        Args:
            update_time: Time for policy update
        """
        self.update_times.append(update_time)
        self.total_updates += 1
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary of performance statistics
        """
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        
        # Calculate average times
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        avg_physics_time = np.mean(self.physics_times) if self.physics_times else 0
        avg_update_time = np.mean(self.update_times) if self.update_times else 0
        
        # Calculate steps per second
        steps_per_second = self.total_steps / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate inference percentage
        inference_percent = 100 * avg_inference_time / avg_step_time if avg_step_time > 0 else 0
        
        # Calculate physics percentage
        physics_percent = 100 * avg_physics_time / avg_step_time if avg_step_time > 0 else 0
        
        # Return statistics
        return {
            'elapsed_time': elapsed_time,
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'avg_step_time': avg_step_time,
            'avg_inference_time': avg_inference_time,
            'avg_physics_time': avg_physics_time,
            'avg_update_time': avg_update_time,
            'steps_per_second': steps_per_second,
            'inference_percent': inference_percent,
            'physics_percent': physics_percent
        }
    
    def print_statistics(self) -> None:
        """Print performance statistics."""
        stats = self.get_statistics()
        
        logger.info("Performance Statistics:")
        logger.info(f"  Elapsed time: {stats['elapsed_time']:.2f} seconds")
        logger.info(f"  Total steps: {stats['total_steps']}")
        logger.info(f"  Total updates: {stats['total_updates']}")
        logger.info(f"  Steps per second: {stats['steps_per_second']:.2f}")
        logger.info(f"  Average step time: {stats['avg_step_time']*1000:.2f} ms")
        logger.info(f"  Average inference time: {stats['avg_inference_time']*1000:.2f} ms ({stats['inference_percent']:.1f}%)")
        logger.info(f"  Average physics time: {stats['avg_physics_time']*1000:.2f} ms ({stats['physics_percent']:.1f}%)")
        logger.info(f"  Average update time: {stats['avg_update_time']*1000:.2f} ms")


def simulate_optimal_field(env, field_config: np.ndarray, n_episodes: int = 1,
                        record_trajectories: bool = True, max_particles: Optional[int] = None) -> TrajectoryRecorder:
    """
    Simulate environment with a fixed magnetic field configuration.
    
    Args:
        env: FFRE environment
        field_config: Magnetic field configuration
        n_episodes: Number of episodes to simulate
        record_trajectories: Whether to record trajectories
        max_particles: Maximum number of particles to record
        
    Returns:
        TrajectoryRecorder instance
    """
    # Setup trajectory recorder
    recorder = TrajectoryRecorder(env, max_particles=max_particles) if record_trajectories else None
    
    # Create dummy agent that always returns the given field config
    class DummyAgent:
        def get_action(self, state, deterministic=True):
            return field_config, 0.0
    
    dummy_agent = DummyAgent()
    
    # Simulate episodes
    for episode in range(n_episodes):
        if record_trajectories:
            episode_info = recorder.record_episode(dummy_agent)
        else:
            # Reset environment
            state, _ = env.reset()
            
            # Run episode
            done = False
            step = 0
            
            while not done and step < getattr(env, 'max_steps', 1000):
                # Take step with fixed field config
                next_state, reward, terminated, truncated, info = env.step(field_config)
                done = terminated or truncated
                
                # Update state
                state = next_state
                step += 1
        
        # Log episode results
        if record_trajectories:
            logger.info(f"Simulation Episode {episode}: "
                      f"Reward={episode_info['reward']:.3f}, "
                      f"Thrust={episode_info['thrust']:.3e}, "
                      f"Efficiency={episode_info['efficiency']:.3e}, "
                      f"Escape Rate={episode_info['escape_rate']:.3f}")
    
    return recorder


def analyze_parameter_sensitivity(env, agent, parameter_name: str, parameter_values: List[float],
                               num_episodes: int = 5) -> Dict[str, List[float]]:
    """
    Analyze sensitivity to a specific environment parameter.
    
    Args:
        env: FFRE environment
        agent: Trained RL agent
        parameter_name: Name of parameter to vary
        parameter_values: List of parameter values to test
        num_episodes: Number of episodes per parameter value
        
    Returns:
        Dictionary of metrics for each parameter value
    """
    # Check if parameter exists in environment
    if not hasattr(env.config, parameter_name) and not (
        hasattr(env.config, 'particle_config') and hasattr(env.config.particle_config, parameter_name)):
        raise ValueError(f"Parameter '{parameter_name}' not found in environment configuration")
    
    # Prepare results
    results = {
        'parameter_values': parameter_values,
        'rewards': [],
        'thrust_values': [],
        'efficiency_values': [],
        'escape_rates': [],
        'wall_collision_rates': []
    }
    
    # Original parameter value
    if hasattr(env.config, parameter_name):
        original_value = getattr(env.config, parameter_name)
        set_param = lambda val: setattr(env.config, parameter_name, val)
    else:
        original_value = getattr(env.config.particle_config, parameter_name)
        set_param = lambda val: setattr(env.config.particle_config, parameter_name, val)
    
    # Test each parameter value
    for value in parameter_values:
        logger.info(f"Testing {parameter_name}={value}")
        
        # Set parameter value
        set_param(value)
        
        # Evaluate agent
        metrics, _, _ = evaluate_agent(agent, env, num_episodes=num_episodes, record_trajectories=False)
        
        # Store results
        results['rewards'].append(np.mean(metrics['rewards']))
        results['thrust_values'].append(np.mean(metrics['thrust_values']))
        results['efficiency_values'].append(np.mean(metrics['efficiency_values']))
        results['escape_rates'].append(np.mean(metrics['escape_rates']))
        results['wall_collision_rates'].append(np.mean(metrics['wall_collision_rates']))
    
    # Restore original value
    set_param(original_value)
    
    return results


def find_optimal_field_configuration(env, field_shape: str = 'linear', 
                                  n_points: int = 10, n_episodes: int = 3) -> np.ndarray:
    """
    Search for optimal field configuration using parameterized shapes.
    
    Args:
        env: FFRE environment
        field_shape: Shape of field ('linear', 'exponential', 'gaussian', 'sigmoid')
        n_points: Number of points to try per parameter
        n_episodes: Number of episodes per point
        
    Returns:
        Optimal field configuration
    """
    # Number of coils
    n_coils = getattr(env.config, 'n_field_coils', 5)
    
    # Create dummy agent class
    class DummyAgent:
        def __init__(self, field_config):
            self.field_config = field_config
            
        def get_action(self, state, deterministic=True):
            return self.field_config, 0.0
    
    # Define parameter ranges based on field shape
    if field_shape == 'linear':
        # Linear: y = slope * x + intercept
        # Parameters: slope, intercept
        param_ranges = [
            np.linspace(-1.0, 1.0, n_points),  # slope
            np.linspace(0.0, 1.0, n_points)    # intercept
        ]
    elif field_shape == 'exponential':
        # Exponential: y = scale * exp(rate * x)
        # Parameters: scale, rate
        param_ranges = [
            np.linspace(0.1, 1.0, n_points),    # scale
            np.linspace(-5.0, 5.0, n_points)    # rate
        ]
    elif field_shape == 'gaussian':
        # Gaussian: y = amplitude * exp(-(x - mean)^2 / (2 * width^2))
        # Parameters: amplitude, mean, width
        param_ranges = [
            np.linspace(0.1, 1.0, n_points),    # amplitude
            np.linspace(0.0, 1.0, n_points),    # mean
            np.linspace(0.1, 0.5, n_points)     # width
        ]
    elif field_shape == 'sigmoid':
        # Sigmoid: y = scale / (1 + exp(-rate * (x - threshold)))
        # Parameters: scale, rate, threshold
        param_ranges = [
            np.linspace(0.1, 1.0, n_points),    # scale
            np.linspace(5.0, 20.0, n_points),   # rate
            np.linspace(0.2, 0.8, n_points)     # threshold
        ]
    else:
        raise ValueError(f"Unknown field shape: {field_shape}")
    
    # Generate field configurations based on shape
    def generate_field(params):
        x = np.linspace(0, 1, n_coils)
        
        if field_shape == 'linear':
            slope, intercept = params
            y = slope * x + intercept
        elif field_shape == 'exponential':
            scale, rate = params
            y = scale * np.exp(rate * x)
        elif field_shape == 'gaussian':
            amplitude, mean, width = params
            y = amplitude * np.exp(-((x - mean) ** 2) / (2 * width ** 2))
        elif field_shape == 'sigmoid':
            scale, rate, threshold = params
            y = scale / (1 + np.exp(-rate * (x - threshold)))
        
        # Clip to [0, 1]
        return np.clip(y, 0, 1)
    
    # Track best configuration
    best_reward = -np.inf
    best_params = None
    best_field_config = None
    
    # Grid search over parameters
    total_combinations = np.prod([len(r) for r in param_ranges])
    combination_count = 0
    
    # Iterate over all parameter combinations
    logger.info(f"Searching {total_combinations} field configurations with shape '{field_shape}'")
    
    # Iterator function for parameter combinations
    def param_combinations(ranges, current=[]):
        if len(ranges) == 0:
            yield current
        else:
            for val in ranges[0]:
                yield from param_combinations(ranges[1:], current + [val])
    
    # Search all combinations
    for params in param_combinations(param_ranges):
        # Generate field configuration
        field_config = generate_field(params)
        
        # Create agent with this field
        agent = DummyAgent(field_config)
        
        # Evaluate
        metrics, _, _ = evaluate_agent(agent, env, num_episodes=n_episodes, record_trajectories=False)
        
        # Calculate average reward
        avg_reward = np.mean(metrics['rewards'])
        
        # Update best if needed
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = params
            best_field_config = field_config
        
        # Update counter
        combination_count += 1
        if combination_count % 10 == 0:
            logger.info(f"Processed {combination_count}/{total_combinations} combinations. Current best reward: {best_reward:.3f}")
    
    # Log best configuration
    logger.info(f"Best {field_shape} configuration: params={best_params}, reward={best_reward:.3f}")
    
    return best_field_config
"""
Evaluation utilities for FFRE optimization.

This module provides tools for evaluating and analyzing trained policies
for FFRE magnetic field optimization.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import time
import os
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)


class TrajectoryRecorder:
    """
    Records particle trajectories for visualization and analysis.
    
    This class provides methods for recording and analyzing particle trajectories
    during FFRE simulation, enabling detailed evaluation of policy performance.
    """
    
    def __init__(self, env, max_particles: Optional[int] = None, 
               record_fields: bool = True):
        """
        Initialize the trajectory recorder.
        
        Args:
            env: FFRE environment
            max_particles: Maximum number of particles to record (None for all)
            record_fields: Whether to record field values along trajectories
        """
        self.env = env
        self.max_particles = max_particles
        self.record_fields = record_fields
        self.trajectories = []
        self.field_data = []
        
        # Statistics
        self.statistics = {}
    
    def record_episode(self, agent, deterministic: bool = True) -> Dict[str, Any]:
        """
        Record particle trajectories for a full episode.
        
        Args:
            agent: RL agent
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary of episode information
        """
        # Reset environment and trajectories
        state, _ = self.env.reset()
        self.trajectories = []
        self.field_data = []
        
        # Get number of particles to record
        if self.max_particles is None:
            if hasattr(self.env, 'config') and hasattr(self.env.config, 'particle_config'):
                n_particles = self.env.config.particle_config.num_particles
            else:
                n_particles = len(self.env.get_all_particles())
        else:
            n_particles = min(self.max_particles, len(self.env.get_all_particles()))
        
        # Initialize trajectory tracking for each particle
        particle_trajectories = [[] for _ in range(n_particles)]
        
        # Initialize field data if recording
        if self.record_fields:
            field_data = [[] for _ in range(n_particles)]
            # Get field function from environment
            if hasattr(self.env, 'get_field_function'):
                field_function = self.env.get_field_function()
            else:
                # Default to zero field if not available
                field_function = lambda pos: np.zeros(3)
        
        # Record initial states
        particles = self.env.get_all_particles()
        for i in range(n_particles):
            if i < len(particles):
                # Record initial state
                particle_trajectories[i].append({
                    'position': particles[i]['position'].copy(),
                    'velocity': particles[i]['velocity'].copy(),
                    'mass': particles[i]['mass'],
                    'charge': particles[i]['charge'],
                    'active': particles[i]['active'],
                    'escaped_thrust': particles[i]['escaped_thrust'],
                    'step': 0
                })
                
                # Record field if enabled
                if self.record_fields:
                    field = field_function(particles[i]['position'])
                    field_data[i].append({
                        'position': particles[i]['position'].copy(),
                        'field': field.copy(),
                        'field_magnitude': np.linalg.norm(field),
                        'step': 0
                    })
        
        # Run episode
        episode_reward = 0
        done = False
        step = 0
        info = {}
        
        while not done and step < getattr(self.env, 'max_steps', 1000):
            # Get action from agent
            action, _ = agent.get_action(state, deterministic=deterministic)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Record particle states
            particles = self.env.get_all_particles()
            for i in range(n_particles):
                if i < len(particles) and particles[i]['active']:
                    # Record state
                    particle_trajectories[i].append({
                        'position': particles[i]['position'].copy(),
                        'velocity': particles[i]['velocity'].copy(),
                        'mass': particles[i]['mass'],
                        'charge': particles[i]['charge'],
                        'active': particles[i]['active'],
                        'escaped_thrust': particles[i]['escaped_thrust'],
                        'step': step + 1
                    })
                    
                    # Record field if enabled
                    if self.record_fields:
                        field = field_function(particles[i]['position'])
                        field_data[i].append({
                            'position': particles[i]['position'].copy(),
                            'field': field.copy(),
                            'field_magnitude': np.linalg.norm(field),
                            'step': step + 1
                        })
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step += 1
        
        # Store trajectories (only those with at least 2 points)
        self.trajectories = [traj for traj in particle_trajectories if len(traj) >= 2]
        
        # Store field data if recorded
        if self.record_fields:
            self.field_data = [data for data in field_data if len(data) >= 2]
        
        # Compute statistics
        self._compute_statistics(info, action)
        
        # Return episode info
        return {
            'reward': episode_reward,
            'steps': step,
            'thrust': info.get('thrust', 0),
            'efficiency': info.get('efficiency', 0),
            'escape_rate': info.get('escape_rate', 0),
            'wall_collisions': info.get('wall_collisions', 0),
            'active_particles': info.get('active_particles', 0),
            'final_field_config': action
        }
    
    def _compute_statistics(self, info: Dict[str, Any], final_field_config: np.ndarray) -> None:
        """
        Compute statistics from recorded trajectories.
        
        Args:
            info: Info dictionary from environment
            final_field_config: Final magnetic field configuration
        """
        # Basic statistics from info
        self.statistics['thrust'] = info.get('thrust', 0)
        self.statistics['efficiency'] = info.get('efficiency', 0)
        self.statistics['escape_rate'] = info.get('escape_rate', 0)
        self.statistics['wall_collisions'] = info.get('wall_collisions', 0)
        self.statistics['active_particles'] = info.get('active_particles', 0)
        self.statistics['final_field_config'] = final_field_config.copy()
        
        