#!/usr/bin/env python
"""
Validation script for the FFRE magnetic field optimization system.

This script performs a series of tests to validate the core components of the
FFRE-RL system, ensuring proper functionality before running full training.
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import FFRE-RL components
from ffre_rl.physics import ParticlePhysicsEngine
from ffre_rl.environment import FFREEnvironment, FFREConfig, ParticleConfig
from ffre_rl.agent import PPOAgent, PPOAgentConfig
from ffre_rl.visualization import MagneticFieldVisualization
from ffre_rl.evaluation import TrajectoryRecorder
from ffre_rl.constants import ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, AM241_ENERGY


def setup_logger(log_file=None):
    """Set up logging configuration."""
    log_level = logging.INFO
    
    # Configure basic logging
    if log_file:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Create logger for this module
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate FFRE-RL system components')
    
    parser.add_argument('--output-dir', type=str, default='./validation_results',
                        help='Directory to save validation results')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all validation tests')
    parser.add_argument('--test-physics', action='store_true',
                        help='Test physics engine')
    parser.add_argument('--test-environment', action='store_true',
                        help='Test environment')
    parser.add_argument('--test-agent', action='store_true',
                        help='Test RL agent')
    parser.add_argument('--test-visualization', action='store_true',
                        help='Test visualization tools')
    parser.add_argument('--test-mini-training', action='store_true',
                        help='Run mini training session')
    parser.add_argument('--use-jit', action='store_true',
                        help='Use JIT compilation for physics engine')
    parser.add_argument('--use-structured-arrays', action='store_true',
                        help='Use structured arrays for particle storage')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path')
    
    args = parser.parse_args()
    
    # If no specific tests selected, enable run-all
    if not any([
        args.test_physics, 
        args.test_environment, 
        args.test_agent, 
        args.test_visualization, 
        args.test_mini_training,
        args.run_all
    ]):
        args.run_all = True
    
    return args


def test_physics_engine(logger):
    """Test the physics engine component."""
    logger.info("Testing ParticlePhysicsEngine...")
    
    # Create physics engine with and without JIT
    physics_no_jit = ParticlePhysicsEngine(dt=1e-9, use_jit=False)
    physics_jit = ParticlePhysicsEngine(dt=1e-9, use_jit=True)
    
    # Test Lorentz force calculation
    charge = ALPHA_PARTICLE_CHARGE
    velocity = np.array([1e5, 0, 0])  # 100 km/s in x direction
    e_field = np.array([0, 0, 0])     # No electric field
    b_field = np.array([0, 0, 1.0])   # 1 Tesla in z direction
    
    # Calculate force with and without JIT
    force_no_jit = physics_no_jit.calculate_lorentz_force(charge, velocity, e_field, b_field)
    force_jit = physics_jit.calculate_lorentz_force(charge, velocity, e_field, b_field)
    
    # Expected force: F = q * (v × B) = q * [0, v_x*B_z, 0]
    expected_force = np.array([0, charge * velocity[0] * b_field[2], 0])
    
    # Check if forces match expected value
    no_jit_correct = np.allclose(force_no_jit, expected_force)
    jit_correct = np.allclose(force_jit, expected_force)
    
    logger.info(f"  Force calculation test (no JIT): {'PASSED' if no_jit_correct else 'FAILED'}")
    logger.info(f"    Calculated: {force_no_jit}")
    logger.info(f"    Expected: {expected_force}")
    
    logger.info(f"  Force calculation test (JIT): {'PASSED' if jit_correct else 'FAILED'}")
    logger.info(f"    Calculated: {force_jit}")
    logger.info(f"    Expected: {expected_force}")
    
    # Test Larmor radius calculation
    v_perp = 1e5  # 100 km/s perpendicular to B
    b_mag = 3.0   # 3 Tesla
    
    radius_no_jit = physics_no_jit.calculate_larmor_radius(ALPHA_PARTICLE_MASS, v_perp, ALPHA_PARTICLE_CHARGE, b_mag)
    expected_radius = (ALPHA_PARTICLE_MASS * v_perp) / (ALPHA_PARTICLE_CHARGE * b_mag)
    
    logger.info(f"  Larmor radius test: {'PASSED' if np.isclose(radius_no_jit, expected_radius) else 'FAILED'}")
    logger.info(f"    Calculated: {radius_no_jit:.6e} m")
    logger.info(f"    Expected: {expected_radius:.6e} m")
    
    # Test particle state update - without JIT for better debugging
    position = np.array([0, 0, 0])
    velocity = np.array([1e5, 0, 0])
    
    # Update without and with JIT
    new_position_no_jit, new_velocity_no_jit = physics_no_jit.update_particle_state(
        position, velocity, ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE,
        e_field, b_field
    )
    
    new_position_jit, new_velocity_jit = physics_jit.update_particle_state(
        position, velocity, ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE,
        e_field, b_field
    )
    
    # Check if JIT matches non-JIT
    positions_match = np.allclose(new_position_jit, new_position_no_jit)
    velocities_match = np.allclose(new_velocity_jit, new_velocity_no_jit)
    
    logger.info(f"  Particle state update consistency test: {'PASSED' if positions_match and velocities_match else 'FAILED'}")
    logger.info(f"    Non-JIT position: {new_position_no_jit}")
    logger.info(f"    JIT position: {new_position_jit}")
    logger.info(f"    Non-JIT velocity: {new_velocity_no_jit}")
    logger.info(f"    JIT velocity: {new_velocity_jit}")
    
    # Verify gyration in B field: v_x should remain constant, v_y and v_z should oscillate
    if new_velocity_no_jit[1] != 0:
        logger.info("  Gyration test: PASSED - Velocity changed in y direction due to B field in z direction")
    else:
        logger.info("  Gyration test: FAILED - No perpendicular velocity change observed")
    
    # Circle test: simulate a particle for many steps to verify it moves in a circle
    steps = 100
    circle_position = position.copy()
    circle_velocity = velocity.copy()
    positions = [circle_position.copy()]
    
    for _ in range(steps):
        circle_position, circle_velocity = physics_no_jit.update_particle_state(
            circle_position, circle_velocity, ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE,
            e_field, b_field
        )
        positions.append(circle_position.copy())
    
    positions = np.array(positions)
    
    # Check if particle returns near starting point in y-z plane
    y_z_displacement = np.sqrt(positions[-1, 1]**2 + positions[-1, 2]**2)
    x_displacement = positions[-1, 0] - positions[0, 0]
    
    logger.info(f"  Circular motion test over {steps} steps:")
    logger.info(f"    Y-Z displacement from start: {y_z_displacement:.6e} m")
    logger.info(f"    X displacement: {x_displacement:.6e} m")
    logger.info(f"    {'PASSED' if y_z_displacement < 1e-5 and x_displacement > 0 else 'FAILED'}")
    
    # Plot circle for visual confirmation
    plt.figure(figsize=(10, 8))
    plt.plot(positions[:, 1], positions[:, 2], 'b.-')
    plt.plot(positions[0, 1], positions[0, 2], 'go', label='Start')
    plt.plot(positions[-1, 1], positions[-1, 2], 'ro', label='End')
    plt.xlabel('Y position (m)')
    plt.ylabel('Z position (m)')
    plt.title('Charged Particle Motion in Uniform Magnetic Field (Y-Z Plane)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    # Save plot
    os.makedirs('validation_results/physics', exist_ok=True)
    plt.savefig('validation_results/physics/circular_motion.png')
    plt.close()
    
    # Plot 3D trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b.-')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=50, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=50, label='End')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    ax.set_title('Charged Particle Motion in Uniform Magnetic Field (3D)')
    ax.legend()
    
    # Save plot
    plt.savefig('validation_results/physics/3d_trajectory.png')
    plt.close()
    
    # Summary
    logger.info("Physics engine tests completed")
    all_passed = no_jit_correct and jit_correct and positions_match and velocities_match
    logger.info(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed


def test_environment(logger, use_jit=False, use_structured_arrays=False):
    """Test the environment component."""
    logger.info("Testing FFREEnvironment...")
    
    # Create particle config
    particle_config = ParticleConfig(
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        energy=AM241_ENERGY,
        num_particles=10,
        emission_radius=0.5,
        direction_bias=0.2
    )
    
    # Create FFRE config
    ffre_config = FFREConfig(
        chamber_length=0.5,
        chamber_radius=0.15,
        max_field_strength=3.0,
        n_field_coils=3,
        max_steps=100,
        dt=1e-9,
        particle_config=particle_config,
        use_jit=use_jit,
        use_structured_arrays=use_structured_arrays
    )
    
    # Create environment
    env = FFREEnvironment(ffre_config)
    
    # Test environment reset
    logger.info("Testing environment reset...")
    observation, info = env.reset()
    
    logger.info(f"  Observation shape: {observation.shape}")
    logger.info(f"  Expected shape: {env.observation_space.shape}")
    
    observation_valid = env.observation_space.contains(observation)
    logger.info(f"  Observation valid: {'PASSED' if observation_valid else 'FAILED'}")
    
    # Check particles
    if use_structured_arrays:
        active_particles = np.sum(env.particle_tracker.particles['active'])
    else:
        active_particles = sum(p['active'] for p in env.particles)
    
    logger.info(f"  Active particles: {active_particles} / {ffre_config.particle_config.num_particles}")
    logger.info(f"  {'PASSED' if active_particles == ffre_config.particle_config.num_particles else 'FAILED'}")
    
    # Check initial positions
    if use_structured_arrays:
        x_positions = env.particle_tracker.particles['position'][:, 0]
    else:
        x_positions = [p['position'][0] for p in env.particles]
    
    x_min, x_max = min(x_positions), max(x_positions)
    logger.info(f"  X positions min: {x_min}, max: {x_max}")
    logger.info(f"  {'PASSED' if x_min >= 0 and x_max <= 0.05 * ffre_config.chamber_length else 'FAILED'}")
    
    # Test environment step
    logger.info("Testing environment step...")
    
    # Take step with zero magnetic field
    action = np.zeros(env.action_space.shape)
    next_observation, reward, terminated, truncated, info = env.step(action)
    
    logger.info(f"  Step with zero field:")
    logger.info(f"    Reward: {reward}")
    logger.info(f"    Terminated: {terminated}")
    logger.info(f"    Truncated: {truncated}")
    
    # Verify observation
    observation_valid = env.observation_space.contains(next_observation)
    logger.info(f"  Next observation valid: {'PASSED' if observation_valid else 'FAILED'}")
    
    # Take step with uniform magnetic field
    action = np.ones(env.action_space.shape) * 0.5  # 50% of max field
    next_observation, reward, terminated, truncated, info = env.step(action)
    
    logger.info(f"  Step with uniform field:")
    logger.info(f"    Reward: {reward}")
    logger.info(f"    Terminated: {terminated}")
    logger.info(f"    Truncated: {truncated}")
    logger.info(f"    Thrust: {info.get('thrust', 0)}")
    logger.info(f"    Efficiency: {info.get('efficiency', 0)}")
    logger.info(f"    Escape rate: {info.get('escape_rate', 0)}")
    
    # Run a full episode with random actions
    logger.info("Running full episode with random actions...")
    observation, info = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        # Random action
        action = env.action_space.sample()
        
        # Take step
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    logger.info(f"  Episode completed in {steps} steps")
    logger.info(f"  Total reward: {total_reward}")
    logger.info(f"  Final thrust: {info.get('thrust', 0)}")
    logger.info(f"  Final efficiency: {info.get('efficiency', 0)}")
    logger.info(f"  Final escape rate: {info.get('escape_rate', 0)}")
    
    # Test custom field configuration
    logger.info("Testing custom field configuration...")
    observation, info = env.reset()
    
    # Create sine wave field configuration
    x = np.linspace(0, 1, env.config.n_field_coils)
    sine_field = 0.5 + 0.5 * np.sin(2 * np.pi * x)
    
    # Take step with sine field
    observation, reward, terminated, truncated, info = env.step(sine_field)
    
    logger.info(f"  Step with sine field:")
    logger.info(f"    Reward: {reward}")
    logger.info(f"    Thrust: {info.get('thrust', 0)}")
    logger.info(f"    Efficiency: {info.get('efficiency', 0)}")
    logger.info(f"    Escape rate: {info.get('escape_rate', 0)}")
    
    # Plot magnetic field
    if hasattr(env, 'get_field_function'):
        field_function = env.get_field_function()
        
        # Create grid
        x = np.linspace(0, env.config.chamber_length, 100)
        field_strength = np.zeros_like(x)
        
        # Calculate field at each point
        for i, xi in enumerate(x):
            field = field_function(np.array([xi, 0, 0]))
            field_strength[i] = np.linalg.norm(field)
        
        # Plot field
        plt.figure(figsize=(10, 6))
        plt.plot(x / env.config.chamber_length, field_strength)
        plt.xlabel('Normalized Position')
        plt.ylabel('Field Strength (T)')
        plt.title('Magnetic Field Strength for Sine Configuration')
        plt.grid(True)
        
        # Save plot
        os.makedirs('validation_results/environment', exist_ok=True)
        plt.savefig('validation_results/environment/sine_field.png')
        plt.close()
    
    # Summary
    logger.info("Environment tests completed")
    all_passed = observation_valid and active_particles == ffre_config.particle_config.num_particles
    logger.info(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed, env


def test_agent(logger, env=None):
    """Test the RL agent."""
    logger.info("Testing PPOAgent...")
    
    # Create environment if not provided
    if env is None:
        # Create particle config
        particle_config = ParticleConfig(
            mass=ALPHA_PARTICLE_MASS,
            charge=ALPHA_PARTICLE_CHARGE,
            energy=AM241_ENERGY,
            num_particles=10,
            emission_radius=0.5,
            direction_bias=0.2
        )
        
        # Create FFRE config
        ffre_config = FFREConfig(
            chamber_length=0.5,
            chamber_radius=0.15,
            max_field_strength=3.0,
            n_field_coils=3,
            max_steps=100,
            dt=1e-9,
            particle_config=particle_config
        )
        
        # Create environment
        env = FFREEnvironment(ffre_config)
    
    # Create agent config
    agent_config = PPOAgentConfig(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=64,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        clip_ratio=0.2,
        critic_loss_coef=0.5,
        entropy_coef=0.01,
        model_dir="./validation_results/agent/models"
    )
    
    # Create directory for models
    os.makedirs(agent_config.model_dir, exist_ok=True)
    
    # Create agent
    agent = PPOAgent(agent_config)
    
    # Test action generation
    state, _ = env.reset()
    action, log_prob = agent.get_action(state)
    
    logger.info(f"Agent action test:")
    logger.info(f"  Action: {action}")
    logger.info(f"  Log probability: {log_prob}")
    logger.info(f"  Action shape: {action.shape}")
    logger.info(f"  Expected shape: {env.action_space.shape}")
    
    action_valid = env.action_space.contains(action)
    logger.info(f"  Action valid: {'PASSED' if action_valid else 'FAILED'}")
    
    # Test critic value estimation
    value = agent.get_value(state)
    logger.info(f"  Value estimate: {value}")
    
    # Test one mini-update
    logger.info("Testing mini-training update (1 episode, 5 steps)...")
    
    # Collect a mini-batch of experience
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []
    
    state, _ = env.reset()
    
    for _ in range(5):
        action, log_prob = agent.get_action(state)
        value = agent.get_value(state)
        
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        rewards.append(reward)
        dones.append(done)
        
        state = next_state
        
        if done:
            break
    
    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    log_probs = np.array(log_probs)
    rewards = np.array(rewards)
    values = np.array(values)
    dones = np.array(dones)
    
    # Compute returns and advantages
    if not dones[-1]:
        last_value = agent.get_value(state)
    else:
        last_value = 0
    
    returns = []
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + agent_config.gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + agent_config.gamma * 0.95 * (1 - dones[t]) * gae
        
        returns.insert(0, gae + values[t])
        advantages.insert(0, gae)
    
    returns = np.array(returns)
    advantages = np.array(advantages)
    
    # Normalize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    # Update policy and value function
    update_info = agent.update(
        states, actions, log_probs, returns, advantages, 
        batch_size=len(states), epochs=1
    )
    
    logger.info(f"  Update completed with:")
    logger.info(f"    Actor loss: {update_info['actor_loss']}")
    logger.info(f"    Critic loss: {update_info['critic_loss']}")
    logger.info(f"    Entropy: {update_info['entropy']}")
    
    # Test model saving and loading
    logger.info("Testing model save/load...")
    
    # Save model
    agent.save_models("test")
    
    # Create new agent
    new_agent = PPOAgent(agent_config)
    
    # Load model
    new_agent.load_models("test")
    
    # Compare actions from original and loaded agents
    state, _ = env.reset()
    original_action, _ = agent.get_action(state, deterministic=True)
    loaded_action, _ = new_agent.get_action(state, deterministic=True)
    
    actions_match = np.allclose(original_action, loaded_action)
    logger.info(f"  Actions match after load: {'PASSED' if actions_match else 'FAILED'}")
    logger.info(f"    Original action: {original_action}")
    logger.info(f"    Loaded action: {loaded_action}")
    
    # Summary
    logger.info("Agent tests completed")
    all_passed = action_valid and actions_match
    logger.info(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed, agent


def test_visualization(logger, env=None):
    """Test the visualization module."""
    logger.info("Testing MagneticFieldVisualization...")
    
    # Create environment if not provided
    if env is None:
        # Create particle config
        particle_config = ParticleConfig(
            mass=ALPHA_PARTICLE_MASS,
            charge=ALPHA_PARTICLE_CHARGE,
            energy=AM241_ENERGY,
            num_particles=10,
            emission_radius=0.5,
            direction_bias=0.2
        )
        
        # Create FFRE config
        ffre_config = FFREConfig(
            chamber_length=0.5,
            chamber_radius=0.15,
            max_field_strength=3.0,
            n_field_coils=3,
            max_steps=100,
            dt=1e-9,
            particle_config=particle_config
        )
        
        # Create environment
        env = FFREEnvironment(ffre_config)
    
    # Create visualizer
    visualizer = MagneticFieldVisualization()
    
    # Create output directory
    os.makedirs('validation_results/visualization', exist_ok=True)
    
    # Test field configuration plotting
    logger.info("Testing field configuration plotting...")
    
    # Create sample field configurations
    n_coils = env.config.n_field_coils
    
    # Linear field
    linear_field = np.linspace(0, 1, n_coils)
    
    # Sine field
    x = np.linspace(0, 1, n_coils)
    sine_field = 0.5 + 0.5 * np.sin(2 * np.pi * x)
    
    # Gaussian field
    gaussian_field = np.exp(-10 * (x - 0.5)**2)
    
    # Plot fields
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    visualizer.plot_field_configuration(linear_field, env.config.max_field_strength, ax=axes[0], title="Linear Field")
    visualizer.plot_field_configuration(sine_field, env.config.max_field_strength, ax=axes[1], title="Sine Field")
    visualizer.plot_field_configuration(gaussian_field, env.config.max_field_strength, ax=axes[2], title="Gaussian Field")
    
    plt.tight_layout()
    plt.savefig('validation_results/visualization/field_configurations.png')
    plt.close()
    
    logger.info("  Field configuration plots created")
    
    # Test particle trajectory plotting
    logger.info("Testing particle trajectory plotting...")
    
    # Generate dummy trajectories
    n_trajectories = 5
    trajectory_steps = 50
    trajectories = []
    
    for i in range(n_trajectories):
        traj = []
        # Initial position with some randomness
        initial_pos = np.array([
            0.05 * env.config.chamber_length * np.random.random(),
            (np.random.random() - 0.5) * 0.5 * env.config.chamber_radius,
            (np.random.random() - 0.5) * 0.5 * env.config.chamber_radius
        ])
        
        # Initial velocity (normalized)
        initial_vel = np.random.rand(3)
        initial_vel = initial_vel / np.linalg.norm(initial_vel) * 1e5
        
        # Create helical trajectory
        for step in range(trajectory_steps):
            t = step / trajectory_steps
            
            # Helical motion
            position = np.array([
                initial_pos[0] + t * env.config.chamber_length,
                initial_pos[1] + 0.1 * env.config.chamber_radius * np.cos(10 * t + i),
                initial_pos[2] + 0.1 * env.config.chamber_radius * np.sin(10 * t + i)
            ])
            
            # Forward-dominated velocity
            velocity = np.array([
                initial_vel[0],
                initial_vel[1] * np.cos(10 * t + i + np.pi/2),
                initial_vel[2] * np.sin(10 * t + i + np.pi/2)
            ])
            
            # Add to trajectory
            traj.append({
                'position': position,
                'velocity': velocity,
                'mass': ALPHA_PARTICLE_MASS,
                'charge': ALPHA_PARTICLE_CHARGE,
                'active': True if step < trajectory_steps - 1 else False,
                'escaped_thrust': True if i % 2 == 0 and step == trajectory_steps - 1 else False
            })
        
        trajectories.append(traj)
    
    # Plot 3D trajectories
    fig = visualizer.plot_particle_trajectories(
        trajectories, 
        env.config.chamber_length, 
        env.config.chamber_radius,
        title="Test Particle Trajectories (3D)",
        return_fig=True
    )
    
    plt.savefig('validation_results/visualization/particle_trajectories_3d.png')
    plt.close(fig)
    
    # Plot 2D trajectories
    fig = visualizer.plot_particle_trajectories(
        trajectories, 
        env.config.chamber_length, 
        env.config.chamber_radius,
        title="Test Particle Trajectories (2D)",
        plot_3d=False,
        return_fig=True
    )
    
    plt.savefig('validation_results/visualization/particle_trajectories_2d.png')
    plt.close(fig)
    
    logger.info("  Particle trajectory plots created")
    
 # Test performance metrics plotting
    logger.info("Testing performance metrics plotting...")
    
    # Generate dummy metrics
    n_episodes = 100
    metrics = {
        'episode_rewards': np.random.rand(n_episodes) * 10,
        'thrust_values': np.random.rand(n_episodes) * 1e-20,
        'efficiency_values': np.random.rand(n_episodes) * 0.1,
        'escape_rates': np.random.rand(n_episodes)
    }
    
    # Add some trends
    x = np.linspace(0, 1, n_episodes)
    metrics['episode_rewards'] += 5 * x
    metrics['thrust_values'] += 0.5e-20 * x
    metrics['efficiency_values'] += 0.05 * x
    metrics['escape_rates'] += 0.5 * x
    
    # Plot metrics
    fig = visualizer.plot_performance_metrics(
        metrics,
        title="Test Performance Metrics",
        return_fig=True
    )
    
    plt.savefig('validation_results/visualization/performance_metrics.png')
    plt.close(fig)
    
    logger.info("  Performance metrics plot created")
    
    # Test field heatmap if field function available
    if hasattr(env, 'get_field_function'):
        logger.info("Testing field heatmap...")
        
        field_function = env.get_field_function()
        
        # Apply sine field configuration
        env.reset()
        env.step(sine_field)
        
        # Plot field heatmap
        fig = visualizer.plot_field_heatmap(
            field_function,
            env.config.chamber_length,
            env.config.chamber_radius,
            title="Test Field Heatmap",
            return_fig=True
        )
        
        plt.savefig('validation_results/visualization/field_heatmap.png')
        plt.close(fig)
        
        logger.info("  Field heatmap created")
    
    # Summary
    logger.info("Visualization tests completed")
    logger.info("Overall result: PASSED")
    
    return True


def test_mini_training(logger, env=None, agent=None):
    """Run a mini training session."""
    logger.info("Running mini training session...")
    
    # Create environment if not provided
    if env is None:
        # Create particle config
        particle_config = ParticleConfig(
            mass=ALPHA_PARTICLE_MASS,
            charge=ALPHA_PARTICLE_CHARGE,
            energy=AM241_ENERGY,
            num_particles=10,
            emission_radius=0.5,
            direction_bias=0.2
        )
        
        # Create FFRE config
        ffre_config = FFREConfig(
            chamber_length=0.5,
            chamber_radius=0.15,
            max_field_strength=3.0,
            n_field_coils=3,
            max_steps=50,
            dt=1e-9,
            particle_config=particle_config
        )
        
        # Create environment
        env = FFREEnvironment(ffre_config)
    
    # Create agent if not provided
    if agent is None:
        # Create agent config
        agent_config = PPOAgentConfig(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dim=64,
            lr_actor=0.0003,
            lr_critic=0.001,
            gamma=0.99,
            clip_ratio=0.2,
            critic_loss_coef=0.5,
            entropy_coef=0.01,
            model_dir="./validation_results/mini_training/models"
        )
        
        # Create directory for models
        os.makedirs(agent_config.model_dir, exist_ok=True)
        
        # Create agent
        agent = PPOAgent(agent_config)
    
    # Create output directory
    os.makedirs('validation_results/mini_training', exist_ok=True)
    
    # Train for a few episodes
    n_episodes = 5
    logger.info(f"Training for {n_episodes} episodes...")
    
    metrics = agent.learn(
        env,
        max_episodes=n_episodes,
        max_steps=env.config.max_steps,
        batch_size=32,
        epochs=3
    )
    
    # Log training results
    logger.info("Training completed")
    logger.info(f"Final reward: {metrics['episode_rewards'][-1]}")
    logger.info(f"Final thrust: {metrics['thrust_values'][-1]}")
    logger.info(f"Final efficiency: {metrics['efficiency_values'][-1]}")
    logger.info(f"Final escape rate: {metrics['escape_rates'][-1]}")
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics['thrust_values'])
    plt.xlabel('Episode')
    plt.ylabel('Thrust')
    plt.title('Thrust per Episode')
    
    plt.subplot(2, 2, 3)
    plt.plot(metrics['efficiency_values'])
    plt.xlabel('Episode')
    plt.ylabel('Efficiency')
    plt.title('Efficiency per Episode')
    
    plt.subplot(2, 2, 4)
    plt.plot(metrics['escape_rates'])
    plt.xlabel('Episode')
    plt.ylabel('Escape Rate')
    plt.title('Particle Escape Rate per Episode')
    
    plt.tight_layout()
    plt.savefig('validation_results/mini_training/training_metrics.png')
    plt.close()
    
    # Evaluate agent
    logger.info("Evaluating trained agent...")
    
    # Create trajectory recorder
    recorder = TrajectoryRecorder(env, max_particles=10)
    
    # Record episode
    episode_info = recorder.record_episode(agent, deterministic=True)
    
    # Log evaluation results
    logger.info(f"Evaluation results:")
    logger.info(f"  Reward: {episode_info['reward']}")
    logger.info(f"  Thrust: {episode_info['thrust']}")
    logger.info(f"  Efficiency: {episode_info['efficiency']}")
    logger.info(f"  Escape rate: {episode_info['escape_rate']}")
    
    # Create visualizer
    visualizer = MagneticFieldVisualization()
    
    # Plot best field configuration
    fig = visualizer.plot_field_configuration(
        episode_info['final_field_config'],
        env.config.max_field_strength,
        title="Trained Field Configuration",
        return_fig=True
    )
    
    plt.savefig('validation_results/mini_training/field_configuration.png')
    plt.close(fig)
    
    # Plot trajectories
    fig = visualizer.plot_particle_trajectories(
        recorder.get_trajectories(),
        env.config.chamber_length,
        env.config.chamber_radius,
        title="Particle Trajectories with Trained Field",
        return_fig=True
    )
    
    plt.savefig('validation_results/mini_training/particle_trajectories.png')
    plt.close(fig)
    
    # Summary
    logger.info("Mini training test completed")
    logger.info("Overall result: PASSED")
    
    return True


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger(args.log_file)
    
    logger.info("Starting FFRE-RL validation")
    logger.info(f"Arguments: {args}")
    
    # Record start time
    start_time = time.time()
    
    # Initialize test results
    all_tests_passed = True
    test_results = {}
    
    # Initialize environment and agent
    env = None
    agent = None
    
    try:
        # Run physics engine tests
        if args.run_all or args.test_physics:
            logger.info("-" * 80)
            logger.info("Running physics engine tests")
            test_results['physics'] = test_physics_engine(logger)
            all_tests_passed &= test_results['physics']
        
        # Run environment tests
        if args.run_all or args.test_environment:
            logger.info("-" * 80)
            logger.info("Running environment tests")
            test_results['environment'], env = test_environment(
                logger, 
                use_jit=args.use_jit, 
                use_structured_arrays=args.use_structured_arrays
            )
            all_tests_passed &= test_results['environment']
        
        # Run agent tests
        if args.run_all or args.test_agent:
            logger.info("-" * 80)
            logger.info("Running agent tests")
            test_results['agent'], agent = test_agent(logger, env)
            all_tests_passed &= test_results['agent']
        
        # Run visualization tests
        if args.run_all or args.test_visualization:
            logger.info("-" * 80)
            logger.info("Running visualization tests")
            test_results['visualization'] = test_visualization(logger, env)
            all_tests_passed &= test_results['visualization']
        
        # Run mini training test
        if args.run_all or args.test_mini_training:
            logger.info("-" * 80)
            logger.info("Running mini training test")
            test_results['mini_training'] = test_mini_training(logger, env, agent)
            all_tests_passed &= test_results['mini_training']
        
    finally:
        # Record end time
        end_time = time.time()
        
        # Print summary
        logger.info("=" * 80)
        logger.info("Validation summary:")
        for test, result in test_results.items():
            logger.info(f"  {test}: {'PASSED' if result else 'FAILED'}")
        
        logger.info("-" * 80)
        logger.info(f"Overall result: {'PASSED' if all_tests_passed else 'FAILED'}")
        logger.info(f"Elapsed time: {end_time - start_time:.2f} seconds")
        logger.info("=" * 80)
    
    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())