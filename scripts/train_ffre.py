#!/usr/bin/env python
"""
Train a reinforcement learning agent for FFRE magnetic field optimization.

This script provides a command-line interface for training RL agents to
optimize magnetic field configurations in Fission Fragment Rocket Engines.
"""

import os
import sys
import argparse
import logging
import time
import json
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import FFRE-RL components
from ffre_rl.environment import FFREEnvironment, FFREConfig, ParticleConfig
from ffre_rl.agent import PPOAgent, PPOAgentConfig, QuantizedPPOAgent
from ffre_rl.visualization import MagneticFieldVisualization
from ffre_rl.evaluation import evaluate_agent, TrajectoryRecorder
from ffre_rl.utils import run_training, save_training_results, memory_usage_profiler, setup_mixed_precision, restore_precision
from ffre_rl.constants import ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, AM241_ENERGY, THORIUM232_ENERGY


def setup_logger(log_dir, debug=False):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train FFRE magnetic field optimization model')
    
    # Environment parameters
    parser.add_argument('--chamber-length', type=float, default=0.5,
                        help='Chamber length in meters (default: 0.5)')
    parser.add_argument('--chamber-radius', type=float, default=0.15,
                        help='Chamber radius in meters (default: 0.15)')
    parser.add_argument('--max-field', type=float, default=5.0,
                        help='Maximum magnetic field strength in Tesla (default: 5.0)')
    parser.add_argument('--num-coils', type=int, default=5,
                        help='Number of magnetic field coils (default: 5)')
    
    # Particle parameters
    parser.add_argument('--particles', type=int, default=100,
                        help='Number of particles to simulate (default: 100)')
    parser.add_argument('--particle-type', type=str, choices=['am241', 'th232'], default='am241',
                        help='Type of alpha-emitting particle (default: am241)')
    parser.add_argument('--emission-radius', type=float, default=0.5,
                        help='Normalized emission radius (default: 0.5)')
    parser.add_argument('--direction-bias', type=float, default=0.2,
                        help='Direction bias towards chamber axis (default: 0.2)')
    
    # Agent parameters
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension of neural networks (default: 256)')
    parser.add_argument('--lr-actor', type=float, default=0.0003,
                        help='Learning rate for actor (default: 0.0003)')
    parser.add_argument('--lr-critic', type=float, default=0.001,
                        help='Learning rate for critic (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--use-micrograd', action='store_true',
                        help='Use micro-gradient framework for memory efficiency')
    parser.add_argument('--quantize', action='store_true',
                        help='Use quantized weights for memory efficiency')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train (default: 1000)')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Maximum steps per episode (default: 200)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs per update (default: 10)')
    parser.add_argument('--save-freq', type=int, default=100,
                        help='Frequency of model saving (default: 100)')
    
    # Optimization options
    parser.add_argument('--use-jit', action='store_true',
                        help='Use JIT compilation for physics engine')
    parser.add_argument('--structured-arrays', action='store_true',
                        help='Use structured arrays for memory efficiency')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory to save output (default: ./output)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation (default: 10)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations during training')
    parser.add_argument('--profile-memory', action='store_true',
                        help='Profile memory usage')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds if specified
    if args.seed is not None:
        np.random.seed(args.seed)
        import tensorflow as tf
        tf.random.set_seed(args.seed)
        import random
        random.seed(args.seed)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"ffre_{timestamp}"
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger(output_dir, args.debug)
    logger.info(f"Starting FFRE training experiment: {exp_name}")
    logger.info(f"Arguments: {args}")
    
    # Set up mixed precision if requested
    original_policy = None
    if args.mixed_precision:
        try:
            import tensorflow as tf
            original_policy = setup_mixed_precision()
        except ImportError:
            logger.warning("TensorFlow not available - mixed precision disabled")
    
    # Determine particle energy based on type
    if args.particle_type == 'am241':
        energy = AM241_ENERGY
        logger.info("Using Am-241 alpha particles (5.5 MeV)")
    else:  # th232
        energy = THORIUM232_ENERGY
        logger.info("Using Th-232 alpha particles (4.08 MeV)")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Create particle config
        particle_config = ParticleConfig(
            mass=ALPHA_PARTICLE_MASS,
            charge=ALPHA_PARTICLE_CHARGE,
            energy=energy,
            num_particles=args.particles,
            emission_radius=args.emission_radius,
            direction_bias=args.direction_bias
        )
        
        # Create FFRE config
        ffre_config = FFREConfig(
            chamber_length=args.chamber_length,
            chamber_radius=args.chamber_radius,
            max_field_strength=args.max_field,
            n_field_coils=args.num_coils,
            max_steps=args.max_steps,
            dt=1e-9,
            particle_config=particle_config,
            use_jit=args.use_jit,
            use_structured_arrays=args.structured_arrays
        )
        
        # Create environment
        logger.info("Creating FFRE environment")
        env = FFREEnvironment(ffre_config)
        
        # Create model directory
        model_dir = os.path.join(output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Create agent config
        agent_config = PPOAgentConfig(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dim=args.hidden_dim,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            model_dir=model_dir,
            use_micrograd=args.use_micrograd
        )
        
        # Create agent
        logger.info("Creating PPO agent")
        if args.quantize:
            agent = QuantizedPPOAgent(agent_config, quantization_bits=8)
            logger.info("Using quantized weights (8-bit)")
        else:
            agent = PPOAgent(agent_config)
        
        # Print architecture summary
        logger.info(f"Agent architecture:\n{agent.get_actor_critic_summary()}")
        
        # Profile memory usage if requested
        if args.profile_memory:
            logger.info("Profiling memory usage...")
            mem_stats = memory_usage_profiler(env, agent, n_steps=100)
            logger.info(f"Memory statistics: {mem_stats}")
        
        # Save configuration
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            # Convert non-serializable objects to strings
            config_dict = {
                'ffre_config': {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v 
                              for k, v in vars(ffre_config).items()},
                'agent_config': {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v 
                               for k, v in vars(agent_config).items()},
                'args': vars(args)
            }
            json.dump(config_dict, f, indent=4)
        
        # Train agent
        logger.info("Starting training...")
        train_start_time = time.time()
        
        metrics = agent.learn(
            env,
            max_episodes=args.episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            epochs=args.epochs,
            save_freq=args.save_freq,
            gamma=args.gamma
        )
        
        train_time = time.time() - train_start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Save training results
        save_training_results(metrics, output_dir, agent)
        
        # Evaluate agent
        logger.info(f"Evaluating agent over {args.eval_episodes} episodes...")
        eval_metrics, recorder, best_field_config = evaluate_agent(
            agent, env, num_episodes=args.eval_episodes
        )
        
        # Save evaluation metrics
        np.save(os.path.join(output_dir, 'eval_rewards.npy'), np.array(eval_metrics['rewards']))
        np.save(os.path.join(output_dir, 'eval_thrust.npy'), np.array(eval_metrics['thrust_values']))
        np.save(os.path.join(output_dir, 'eval_efficiency.npy'), np.array(eval_metrics['efficiency_values']))
        np.save(os.path.join(output_dir, 'eval_escape_rates.npy'), np.array(eval_metrics['escape_rates']))
        np.save(os.path.join(output_dir, 'best_field_config.npy'), best_field_config)
        
        # Create visualizer
        if args.visualize:
            logger.info("Generating visualizations...")
            visualizer = MagneticFieldVisualization()
            
            # Plot best field configuration
            fig = visualizer.plot_field_configuration(
                best_field_config, env.config.max_field_strength, return_fig=True)
            fig.savefig(os.path.join(output_dir, 'best_field_config.png'))
            plt.close(fig)
            
            # Plot trajectories
            if recorder is not None:
                fig = visualizer.plot_particle_trajectories(
                    recorder.get_trajectories(), 
                    env.config.chamber_length, 
                    env.config.chamber_radius,
                    return_fig=True
                )
                fig.savefig(os.path.join(output_dir, 'particle_trajectories.png'))
                plt.close(fig)
                
                # Save trajectories
                recorder.save_trajectories(os.path.join(output_dir, 'trajectories.npz'))
            
            # Plot performance metrics
            fig = visualizer.plot_performance_metrics(metrics, return_fig=True)
            fig.savefig(os.path.join(output_dir, 'performance_metrics.png'))
            plt.close(fig)
            
            # Plot field heat map if magnetic field function available
            if hasattr(env, 'get_field_function'):
                field_func = env.get_field_function()
                fig = visualizer.plot_field_heatmap(
                    field_func, 
                    env.config.chamber_length, 
                    env.config.chamber_radius,
                    return_fig=True
                )
                fig.savefig(os.path.join(output_dir, 'field_heatmap.png'))
                plt.close(fig)
        
        # Generate report
        with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
            f.write("FFRE Magnetic Field Optimization Report\n")
            f.write("=====================================\n\n")
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total training time: {train_time:.2f} seconds\n")
            f.write(f"Episodes: {args.episodes}\n")
            f.write(f"Particles simulated: {args.particles}\n")
            f.write(f"Chamber dimensions: {args.chamber_length}m x {args.chamber_radius}m radius\n")
            f.write(f"Number of magnetic field coils: {args.num_coils}\n")
            f.write(f"Maximum field strength: {args.max_field} Tesla\n\n")
            
            f.write("Training Results:\n")
            f.write(f"  Final reward: {metrics['episode_rewards'][-1]:.3f}\n")
            f.write(f"  Final thrust: {metrics['thrust_values'][-1]:.3e}\n")
            f.write(f"  Final efficiency: {metrics['efficiency_values'][-1]:.3e}\n")
            f.write(f"  Final escape rate: {metrics['escape_rates'][-1]:.3f}\n\n")
            
            f.write("Evaluation Results:\n")
            f.write(f"  Average reward: {np.mean(eval_metrics['rewards']):.3f}\n")
            f.write(f"  Average thrust: {np.mean(eval_metrics['thrust_values']):.3e}\n")
            f.write(f"  Average efficiency: {np.mean(eval_metrics['efficiency_values']):.3e}\n")
            f.write(f"  Average escape rate: {np.mean(eval_metrics['escape_rates']):.3f}\n\n")
            
            f.write("Best Field Configuration:\n")
            for i, value in enumerate(best_field_config):
                f.write(f"  Coil {i+1}: {value * args.max_field:.3f} Tesla\n")
    
    finally:
        # Restore precision if needed
        if original_policy is not None:
            restore_precision(original_policy)
        
        # Record total time
        total_time = time.time() - start_time
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()