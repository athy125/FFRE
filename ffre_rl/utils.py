"""
Utility functions for FFRE optimization.

This module provides utility functions for training, evaluating, and running
the FFRE optimization system.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import time
import os
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import json

# Configure logging
logger = logging.getLogger(__name__)


def run_training(params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any, Dict[str, List[float]], Any]:
    """
    Run the complete training pipeline.
    
    Args:
        params: Dictionary of parameters (optional)
        
    Returns:
        Tuple of (agent, env, metrics, visualizer)
    """
    # Lazy import to avoid circular dependencies
    from .environment import FFREEnvironment, FFREConfig, ParticleConfig
    from .agent import PPOAgent, PPOAgentConfig
    from .visualization import MagneticFieldVisualization
    from .constants import ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, AM241_ENERGY
    
    # Default parameters
    default_params = {
        'chamber_length': 0.5,      # 50 cm
        'chamber_radius': 0.15,     # 15 cm
        'num_particles': 100,
        'max_steps': 200,
        'max_field_strength': 5.0,  # Tesla
        'n_field_coils': 5,
        'hidden_dim': 256,
        'lr_actor': 0.0003,
        'lr_critic': 0.001,
        'gamma': 0.99,
        'clip_ratio': 0.2,
        'critic_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'train_episodes': 1000,
        'batch_size': 64,
        'epochs': 10,
        'save_freq': 100,
        'model_dir': './models',
        'use_jit': True,
        'use_structured_arrays': True,
        'use_micrograd': False
    }
    
    # Update with provided parameters
    if params:
        default_params.update(params)
    
    params = default_params
    
    logger.info(f"Starting FFRE magnetic field optimization with parameters: {params}")
    
    # Create particle config
    particle_config = ParticleConfig(
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        energy=AM241_ENERGY,
        num_particles=params['num_particles'],
        emission_radius=0.5,  # Use half of chamber radius
        direction_bias=0.2     # Slight bias toward chamber axis
    )
    
    # Create FFRE config
    ffre_config = FFREConfig(
        chamber_length=params['chamber_length'],
        chamber_radius=params['chamber_radius'],
        max_field_strength=params['max_field_strength'],
        n_field_coils=params['n_field_coils'],
        max_steps=params['max_steps'],
        dt=1e-9,
        particle_config=particle_config,
        use_jit=params['use_jit'],
        use_structured_arrays=params['use_structured_arrays']
    )
    
    # Create environment
    env = FFREEnvironment(ffre_config)
    
    # Create agent config
    agent_config = PPOAgentConfig(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=params['hidden_dim'],
        lr_actor=params['lr_actor'],
        lr_critic=params['lr_critic'],
        gamma=params['gamma'],
        clip_ratio=params['clip_ratio'],
        critic_loss_coef=params['critic_loss_coef'],
        entropy_coef=params['entropy_coef'],
        model_dir=params['model_dir'],
        use_micrograd=params['use_micrograd']
    )
    
    # Create agent
    agent = PPOAgent(agent_config)
    
    # Create visualizer
    visualizer = MagneticFieldVisualization()
    
    # Create model directory if it doesn't exist
    os.makedirs(params['model_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(params['model_dir'], 'config.json'), 'w') as f:
        # Convert non-serializable objects to strings
        config_dict = {
            'ffre_config': {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v 
                          for k, v in vars(ffre_config).items()},
            'agent_config': {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v 
                           for k, v in vars(agent_config).items()},
            'training_params': params
        }
        json.dump(config_dict, f, indent=4)
    
    # Train agent
    logger.info("Starting training...")
    start_time = time.time()
    
    metrics = agent.learn(
        env,
        max_episodes=params['train_episodes'],
        max_steps=params['max_steps'],
        update_freq=params['batch_size'] * 2,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        save_freq=params['save_freq'],
        gamma=params['gamma']
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    return agent, env, metrics, visualizer


def evaluate_agent(agent, env, num_episodes: int = 10) -> Tuple[Dict[str, List[float]], List[List[Dict[str, Any]]], np.ndarray]:
    """
    Evaluate a trained agent.
    
    Args:
        agent: Trained RL agent
        env: FFRE environment
        num_episodes: Number of episodes to evaluate

        Returns:
        Tuple of (evaluation metrics, trajectory data, best field configuration)
    """
    # Lazy import to avoid circular dependencies
    from .evaluation import evaluate_agent as eval_agent
    
    # Evaluate agent
    metrics, recorder, best_field_config = eval_agent(
        agent, env, num_episodes=num_episodes, deterministic=True,
        record_trajectories=True, max_particles=20
    )
    
    # Get trajectories
    trajectories = recorder.get_trajectories() if recorder else []
    
    return metrics, trajectories, best_field_config


def memory_usage_profiler(env, agent, n_steps: int = 100) -> Dict[str, float]:
    """
    Profile memory usage during simulation.
    
    Args:
        env: FFRE environment
        agent: RL agent
        n_steps: Number of steps to simulate
        
    Returns:
        Dictionary of memory usage statistics
    """
    try:
        import psutil
        import gc
    except ImportError:
        logger.warning("psutil not installed. Memory profiling will not be accurate.")
        return {}
    
    # Force garbage collection
    gc.collect()
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Reset environment
    state, _ = env.reset()
    
    # Run simulation
    for _ in range(n_steps):
        # Get action
        action, _ = agent.get_action(state)
        
        # Take step
        next_state, _, done, _, _ = env.step(action)
        
        # Update state
        state = next_state
        
        if done:
            state, _ = env.reset()
    
    # Force garbage collection
    gc.collect()
    
    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Get peak memory usage
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate memory increase
    memory_increase = final_memory - initial_memory
    
    # Return statistics
    return {
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory,
        'peak_memory_mb': peak_memory,
        'memory_increase_mb': memory_increase,
        'memory_per_step_mb': memory_increase / n_steps if n_steps > 0 else 0
    }


def save_training_results(metrics: Dict[str, List[float]], output_dir: str, agent=None, 
                       best_field_config: Optional[np.ndarray] = None) -> None:
    """
    Save training results to disk.
    
    Args:
        metrics: Dictionary of training metrics
        output_dir: Directory to save results
        agent: Trained agent (optional)
        best_field_config: Best field configuration (optional)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    for name, values in metrics.items():
        np.save(os.path.join(output_dir, f"{name}.npy"), np.array(values))
    
    # Save best field configuration if provided
    if best_field_config is not None:
        np.save(os.path.join(output_dir, "best_field_config.npy"), best_field_config)
    
    # Save agent if provided
    if agent is not None:
        agent.save_models("final")
    
    # Save metrics plot
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(metrics.get('episode_rewards', []))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics.get('thrust_values', []))
    plt.xlabel('Episode')
    plt.ylabel('Thrust')
    plt.title('Thrust per Episode')
    
    plt.subplot(2, 2, 3)
    plt.plot(metrics.get('efficiency_values', []))
    plt.xlabel('Episode')
    plt.ylabel('Efficiency')
    plt.title('Efficiency per Episode')
    
    plt.subplot(2, 2, 4)
    plt.plot(metrics.get('escape_rates', []))
    plt.xlabel('Episode')
    plt.ylabel('Escape Rate')
    plt.title('Particle Escape Rate per Episode')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close()
    
    logger.info(f"Training results saved to {output_dir}")


def check_model_compatibility(model_path: str, env) -> Tuple[bool, str]:
    """
    Check if a saved model is compatible with the given environment.
    
    Args:
        model_path: Path to the model directory
        env: FFRE environment to check compatibility with
        
    Returns:
        Tuple of (is_compatible, reason)
    """
    # Check if model exists
    if not os.path.exists(model_path):
        return False, f"Model path {model_path} does not exist"
    
    # Check if config file exists
    config_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_path):
        return False, f"Config file {config_path} does not exist"
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check state dimensions
    model_state_dim = int(config.get('agent_config', {}).get('state_dim', 0))
    env_state_dim = env.observation_space.shape[0]
    
    if model_state_dim != env_state_dim:
        return False, f"State dimension mismatch: model={model_state_dim}, env={env_state_dim}"
    
    # Check action dimensions
    model_action_dim = int(config.get('agent_config', {}).get('action_dim', 0))
    env_action_dim = env.action_space.shape[0]
    
    if model_action_dim != env_action_dim:
        return False, f"Action dimension mismatch: model={model_action_dim}, env={env_action_dim}"
    
    # Check n_field_coils
    model_n_coils = int(config.get('ffre_config', {}).get('n_field_coils', 0))
    env_n_coils = getattr(env.config, 'n_field_coils', 0)
    
    if model_n_coils != env_n_coils:
        return False, f"Number of field coils mismatch: model={model_n_coils}, env={env_n_coils}"
    
    return True, "Model is compatible with environment"


def load_and_evaluate_model(model_path: str, env) -> Tuple[Any, Dict[str, List[float]], np.ndarray]:
    """
    Load a saved model and evaluate it on the given environment.
    
    Args:
        model_path: Path to the model directory
        env: FFRE environment to evaluate on
        
    Returns:
        Tuple of (agent, metrics, best_field_config)
    """
    # Lazy import to avoid circular dependencies
    from .agent import PPOAgent, PPOAgentConfig
    
    # Check model compatibility
    compatible, reason = check_model_compatibility(model_path, env)
    if not compatible:
        raise ValueError(f"Model is not compatible with environment: {reason}")
    
    # Load config
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Create agent config
    agent_config = PPOAgentConfig(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=int(config.get('agent_config', {}).get('hidden_dim', 256)),
        lr_actor=float(config.get('agent_config', {}).get('lr_actor', 0.0003)),
        lr_critic=float(config.get('agent_config', {}).get('lr_critic', 0.001)),
        gamma=float(config.get('agent_config', {}).get('gamma', 0.99)),
        clip_ratio=float(config.get('agent_config', {}).get('clip_ratio', 0.2)),
        critic_loss_coef=float(config.get('agent_config', {}).get('critic_loss_coef', 0.5)),
        entropy_coef=float(config.get('agent_config', {}).get('entropy_coef', 0.01)),
        model_dir=model_path,
        use_micrograd=config.get('agent_config', {}).get('use_micrograd', 'False') == 'True'
    )
    
    # Create agent
    agent = PPOAgent(agent_config)
    
    # Load model
    agent.load_models("final")
    
    # Evaluate agent
    metrics, _, best_field_config = evaluate_agent(agent, env)
    
    return agent, metrics, best_field_config


def compare_models(model_paths: List[str], env, num_episodes: int = 5) -> Dict[str, Dict[str, List[float]]]:
    """
    Compare multiple models on the same environment.
    
    Args:
        model_paths: List of paths to model directories
        env: FFRE environment to evaluate on
        num_episodes: Number of episodes for evaluation
        
    Returns:
        Dictionary of metrics for each model
    """
    # Lazy import to avoid circular dependencies
    from .evaluation import evaluate_agent as eval_agent
    
    # Results dictionary
    results = {}
    
    # Evaluate each model
    for model_path in model_paths:
        logger.info(f"Evaluating model at {model_path}")
        
        # Load model
        agent, _, _ = load_and_evaluate_model(model_path, env)
        
        # Evaluate agent
        metrics, _, _ = eval_agent(
            agent, env, num_episodes=num_episodes, deterministic=True,
            record_trajectories=False
        )
        
        # Extract model name from path
        model_name = os.path.basename(model_path)
        
        # Store results
        results[model_name] = metrics
    
    return results


def optimize_hyperparameters(param_grid: Dict[str, List[Any]], n_trials: int = 5,
                          base_params: Optional[Dict[str, Any]] = None,
                          episodes_per_trial: int = 100) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """
    Run hyperparameter optimization for FFRE RL training.
    
    Args:
        param_grid: Dictionary of parameters and their possible values
        n_trials: Number of random trials
        base_params: Base parameters to update
        episodes_per_trial: Number of episodes per trial
        
    Returns:
        Tuple of (best parameters, metrics for best parameters)
    """
    # Lazy import to avoid circular dependencies
    import random
    
    # Start with default parameters if not provided
    if base_params is None:
        base_params = {
            'chamber_length': 0.5,
            'chamber_radius': 0.15,
            'num_particles': 50,  # Reduced for faster evaluation
            'max_steps': 100,     # Reduced for faster evaluation
            'max_field_strength': 5.0,
            'n_field_coils': 5,
            'hidden_dim': 256,
            'lr_actor': 0.0003,
            'lr_critic': 0.001,
            'gamma': 0.99,
            'clip_ratio': 0.2,
            'critic_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'train_episodes': episodes_per_trial,
            'batch_size': 64,
            'epochs': 10,
            'save_freq': episodes_per_trial,
            'model_dir': './hyperopt_models'
        }
    
    # Track best parameters and performance
    best_params = None
    best_performance = -np.inf
    best_metrics = None
    
    # Create output directory
    os.makedirs(base_params['model_dir'], exist_ok=True)
    
    # Store results for each trial
    all_results = []
    
    # Run trials
    for trial in range(n_trials):
        # Sample parameters from grid
        trial_params = base_params.copy()
        
        for param_name, param_values in param_grid.items():
            trial_params[param_name] = random.choice(param_values)
        
        # Set model directory for this trial
        trial_params['model_dir'] = os.path.join(base_params['model_dir'], f"trial_{trial}")
        
        logger.info(f"Trial {trial}/{n_trials} with parameters: {trial_params}")
        
        # Run training
        agent, env, metrics, _ = run_training(trial_params)
        
        # Evaluate performance (use last 10% of episodes)
        n_last = max(1, len(metrics['episode_rewards']) // 10)
        avg_reward = np.mean(metrics['episode_rewards'][-n_last:])
        avg_thrust = np.mean(metrics['thrust_values'][-n_last:]) if 'thrust_values' in metrics else 0
        avg_efficiency = np.mean(metrics['efficiency_values'][-n_last:]) if 'efficiency_values' in metrics else 0
        avg_escape_rate = np.mean(metrics['escape_rates'][-n_last:]) if 'escape_rates' in metrics else 0
        
        # Compute combined performance metric
        # Weight based on importance
        performance = (
            0.4 * avg_reward +
            0.3 * avg_thrust +
            0.2 * avg_efficiency +
            0.1 * avg_escape_rate
        )
        
        # Store results
        all_results.append({
            'params': trial_params,
            'performance': performance,
            'avg_reward': avg_reward,
            'avg_thrust': avg_thrust,
            'avg_efficiency': avg_efficiency,
            'avg_escape_rate': avg_escape_rate
        })
        
        # Update best if improved
        if performance > best_performance:
            best_performance = performance
            best_params = trial_params.copy()
            best_metrics = metrics
            
            logger.info(f"New best performance: {best_performance:.4f}")
        
        # Save all results
        with open(os.path.join(base_params['model_dir'], 'hyperopt_results.json'), 'w') as f:
            # Convert numpy arrays and non-serializable objects to lists/strings
            serializable_results = []
            for result in all_results:
                serializable_result = {
                    'params': {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v 
                             for k, v in result['params'].items()},
                    'performance': float(result['performance']),
                    'avg_reward': float(result['avg_reward']),
                    'avg_thrust': float(result['avg_thrust']),
                    'avg_efficiency': float(result['avg_efficiency']),
                    'avg_escape_rate': float(result['avg_escape_rate'])
                }
                serializable_results.append(serializable_result)
                
            json.dump(serializable_results, f, indent=4)
    
    # Sort results by performance
    all_results.sort(key=lambda x: x['performance'], reverse=True)
    
    # Print top results
    logger.info("Top hyperparameter configurations:")
    for i, result in enumerate(all_results[:min(3, len(all_results))]):
        logger.info(f"{i+1}. Performance: {result['performance']:.4f}")
        for param_name in param_grid.keys():
            logger.info(f"   {param_name}: {result['params'][param_name]}")
    
    return best_params, best_metrics


def setup_mixed_precision():
    """
    Set up mixed precision training to improve performance.
    
    Returns:
        Original policy function for restoring precision
    """
    try:
        # Set up mixed precision
        original_policy = tf.keras.mixed_precision.global_policy()
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        logger.info(f"Mixed precision enabled with policy: {policy.name}")
        
        return original_policy
    except:
        logger.warning("Failed to set up mixed precision. Continuing with default precision.")
        return None


def restore_precision(original_policy):
    """
    Restore original precision policy.
    
    Args:
        original_policy: Original policy to restore
    """
    if original_policy is not None:
        try:
            tf.keras.mixed_precision.set_global_policy(original_policy)
            logger.info(f"Precision restored to: {original_policy.name}")
        except:
            logger.warning("Failed to restore precision policy.")
"""
Utility functions for FFRE optimization.

This module provides utility functions for training, evaluating, and running
the FFRE optimization system.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import time
import os
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import json

# Configure logging
logger = logging.getLogger(__name__)


def run_training(params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any, Dict[str, List[float]], Any]:
    """
    Run the complete training pipeline.
    
    Args:
        params: Dictionary of parameters (optional)
        
    Returns:
        Tuple of (agent, env, metrics, visualizer)
    """
    # Lazy import to avoid circular dependencies
    from .environment import FFREEnvironment, FFREConfig, ParticleConfig
    from .agent import PPOAgent, PPOAgentConfig
    from .visualization import MagneticFieldVisualization
    from .constants import ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, AM241_ENERGY
    
    # Default parameters
    default_params = {
        'chamber_length': 0.5,      # 50 cm
        'chamber_radius': 0.15,     # 15 cm
        'num_particles': 100,
        'max_steps': 200,
        'max_field_strength': 5.0,  # Tesla
        'n_field_coils': 5,
        'hidden_dim': 256,
        'lr_actor': 0.0003,
        'lr_critic': 0.001,
        'gamma': 0.99,
        'clip_ratio': 0.2,
        'critic_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'train_episodes': 1000,
        'batch_size': 64,
        'epochs': 10,
        'save_freq': 100,
        'model_dir': './models',
        'use_jit': True,
        'use_structured_arrays': True,
        'use_micrograd': False
    }
    
    # Update with provided parameters
    if params:
        default_params.update(params)
    
    params = default_params
    
    logger.info(f"Starting FFRE magnetic field optimization with parameters: {params}")
    
    # Create particle config
    particle_config = ParticleConfig(
        mass=ALPHA_PARTICLE_MASS,
        charge=ALPHA_PARTICLE_CHARGE,
        energy=AM241_ENERGY,
        num_particles=params['num_particles'],
        emission_radius=0.5,  # Use half of chamber radius
        direction_bias=0.2     # Slight bias toward chamber axis
    )
    
    # Create FFRE config
    ffre_config = FFREConfig(
        chamber_length=params['chamber_length'],
        chamber_radius=params['chamber_radius'],
        max_field_strength=params['max_field_strength'],
        n_field_coils=params['n_field_coils'],
        max_steps=params['max_steps'],
        dt=1e-9,
        particle_config=particle_config,
        use_jit=params['use_jit'],
        use_structured_arrays=params['use_structured_arrays']
    )
    
    # Create environment
    env = FFREEnvironment(ffre_config)
    
    # Create agent config
    agent_config = PPOAgentConfig(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=params['hidden_dim'],
        lr_actor=params['lr_actor'],
        lr_critic=params['lr_critic'],
        gamma=params['gamma'],
        clip_ratio=params['clip_ratio'],
        critic_loss_coef=params['critic_loss_coef'],
        entropy_coef=params['entropy_coef'],
        model_dir=params['model_dir'],
        use_micrograd=params['use_micrograd']
    )
    
    # Create agent
    agent = PPOAgent(agent_config)
    
    # Create visualizer
    visualizer = MagneticFieldVisualization()
    
    # Create model directory if it doesn't exist
    os.makedirs(params['model_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(params['model_dir'], 'config.json'), 'w') as f:
        # Convert non-serializable objects to strings
        config_dict = {
            'ffre_config': {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v 
                          for k, v in vars(ffre_config).items()},
            'agent_config': {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v 
                           for k, v in vars(agent_config).items()},
            'training_params': params
        }
        json.dump(config_dict, f, indent=4)
    
    # Train agent
    logger.info("Starting training...")
    start_time = time.time()
    
    metrics = agent.learn(
        env,
        max_episodes=params['train_episodes'],
        max_steps=params['max_steps'],
        update_freq=params['batch_size'] * 2,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        save_freq=params['save_freq'],
        gamma=params['gamma']
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    return agent, env, metrics, visualizer


# def evaluate_agent(agent, env, num_episodes: int = 10) -> Tuple[Dict[str, List[float]], List[List[Dict[str, Any]]], np.ndarray]:
#     """
#     Evaluate a trained agent.
    
#     Args:
#         agent: Trained RL agent
#         env: FFRE environment
#         num_episodes: Number of episodes to evaluate
        
#     Returns:
#         Tuple