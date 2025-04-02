# Load TensorFlow models
            self.actor = keras.models.load_model(f"{self.config.model_dir}/actor_{suffix}")
            self.critic = keras.models.load_model(f"{self.config.model_dir}/critic_{suffix}")
        
        logger.info(f"Models loaded with suffix '{suffix}'")

    def get_actor_critic_summary(self) -> str:
        """
        Get summary of actor and critic networks.
        
        Returns:
            String summary of models
        """
        if self.config.use_micrograd:
            # Summary for micro-gradient models
            actor_params = (
                np.size(self.actor_micro.w1) + 
                np.size(self.actor_micro.b1) + 
                np.size(self.actor_micro.w2) + 
                np.size(self.actor_micro.b2) + 
                np.size(self.actor_micro.w3) + 
                np.size(self.actor_micro.b3)
            )
            
            critic_params = (
                np.size(self.critic_micro.w1) + 
                np.size(self.critic_micro.b1) + 
                np.size(self.critic_micro.w2) + 
                np.size(self.critic_micro.b2) + 
                np.size(self.critic_micro.w3) + 
                np.size(self.critic_micro.b3)
            )
            
            summary = (
                f"Micro-gradient Actor: {actor_params} parameters\n"
                f"Micro-gradient Critic: {critic_params} parameters\n"
                f"Total parameters: {actor_params + critic_params}"
            )
        else:
            # Summary for TensorFlow models
            actor_summary = []
            self.actor.summary(print_fn=lambda x: actor_summary.append(x))
            
            critic_summary = []
            self.critic.summary(print_fn=lambda x: critic_summary.append(x))
            
            summary = (
                f"Actor Summary:\n{''.join(actor_summary)}\n\n"
                f"Critic Summary:\n{''.join(critic_summary)}"
            )
        
        return summary
    
    @staticmethod
    def compute_returns_and_advantages(rewards: List[float], values: List[float], 
                                     dones: List[bool], gamma: float, 
                                     next_state: np.ndarray, done: bool, 
                                     critic: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages for PPO.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            gamma: Discount factor
            next_state: Last observed state
            done: Whether the episode ended
            critic: Critic network
            
        Returns:
            Tuple of returns and advantages arrays
        """
        # Get final value estimate if not done
        if not done:
            if isinstance(critic, MicroGradCritic):
                last_value = critic.forward(next_state)
            else:
                last_value = float(critic(np.expand_dims(next_state, axis=0))[0, 0].numpy())
        else:
            last_value = 0
        
        # Compute returns using the discount factor (GAE - Generalized Advantage Estimation)
        returns = []
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[step + 1]
                
            delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + gamma * 0.95 * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        
        # Compute advantages
        advantages = np.array(returns) - np.array(values)
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return np.array(returns), advantages


class QuantizedPPOAgent(PPOAgent):
    """
    Memory-efficient PPO agent with quantized weights for embedded applications.
    
    This agent uses quantization techniques to reduce memory footprint,
    suitable for resource-constrained environments.
    """
    
    def __init__(self, config: PPOAgentConfig, quantization_bits: int = 8):
        """
        Initialize the quantized PPO agent.
        
        Args:
            config: Configuration object for the agent
            quantization_bits: Number of bits for weight quantization
        """
        super().__init__(config)
        
        # Set quantization parameters
        self.quantization_bits = quantization_bits
        self.quantize_weights = True
        
        # Initialize quantization tables
        self._init_quantization()
    
    def _init_quantization(self) -> None:
        """Initialize quantization tables and parameters."""
        # Determine quantization range
        self.quant_min = -(2 ** (self.quantization_bits - 1))
        self.quant_max = 2 ** (self.quantization_bits - 1) - 1
        
        # Initialize scaling factors
        if self.config.use_micrograd:
            # Quantize micro-gradient models
            self.actor_scales = {
                'w1': self._get_scale(self.actor_micro.w1),
                'b1': self._get_scale(self.actor_micro.b1),
                'w2': self._get_scale(self.actor_micro.w2),
                'b2': self._get_scale(self.actor_micro.b2),
                'w3': self._get_scale(self.actor_micro.w3),
                'b3': self._get_scale(self.actor_micro.b3)
            }
            
            self.critic_scales = {
                'w1': self._get_scale(self.critic_micro.w1),
                'b1': self._get_scale(self.critic_micro.b1),
                'w2': self._get_scale(self.critic_micro.w2),
                'b2': self._get_scale(self.critic_micro.b2),
                'w3': self._get_scale(self.critic_micro.w3),
                'b3': self._get_scale(self.critic_micro.b3)
            }
            
            # Quantize weights
            self._quantize_micro_models()
    
    def _get_scale(self, weights: np.ndarray) -> float:
        """
        Calculate scaling factor for quantization.
        
        Args:
            weights: Weight array
            
        Returns:
            Scaling factor
        """
        # Calculate max absolute value
        max_abs = np.max(np.abs(weights))
        
        # Calculate scale
        if max_abs > 0:
            return max_abs / self.quant_max
        else:
            return 1.0
    
    def _quantize_micro_models(self) -> None:
        """Quantize micro-gradient models."""
        if not self.quantize_weights:
            return
            
        # Quantize actor weights
        self.actor_micro.w1 = self._quantize_array(self.actor_micro.w1, self.actor_scales['w1'])
        self.actor_micro.b1 = self._quantize_array(self.actor_micro.b1, self.actor_scales['b1'])
        self.actor_micro.w2 = self._quantize_array(self.actor_micro.w2, self.actor_scales['w2'])
        self.actor_micro.b2 = self._quantize_array(self.actor_micro.b2, self.actor_scales['b2'])
        self.actor_micro.w3 = self._quantize_array(self.actor_micro.w3, self.actor_scales['w3'])
        self.actor_micro.b3 = self._quantize_array(self.actor_micro.b3, self.actor_scales['b3'])
        
        # Quantize critic weights
        self.critic_micro.w1 = self._quantize_array(self.critic_micro.w1, self.critic_scales['w1'])
        self.critic_micro.b1 = self._quantize_array(self.critic_micro.b1, self.critic_scales['b1'])
        self.critic_micro.w2 = self._quantize_array(self.critic_micro.w2, self.critic_scales['w2'])
        self.critic_micro.b2 = self._quantize_array(self.critic_micro.b2, self.critic_scales['b2'])
        self.critic_micro.w3 = self._quantize_array(self.critic_micro.w3, self.critic_scales['w3'])
        self.critic_micro.b3 = self._quantize_array(self.critic_micro.b3, self.critic_scales['b3'])
    
    def _quantize_array(self, array: np.ndarray, scale: float) -> np.ndarray:
        """
        Quantize array to specified bits.
        
        Args:
            array: Array to quantize
            scale: Scaling factor
            
        Returns:
            Quantized array
        """
        if scale == 0:
            return array
            
        # Quantize to integers
        quant_array = np.round(array / scale).astype(np.int32)
        
        # Clip to quantization range
        quant_array = np.clip(quant_array, self.quant_min, self.quant_max)
        
        # Convert back to float using scale
        return quant_array.astype(np.float32) * scale
    
    def _dequantize_array(self, array: np.ndarray, scale: float) -> np.ndarray:
        """
        Dequantize array to full precision.
        
        Args:
            array: Quantized array
            scale: Scaling factor
            
        Returns:
            Dequantized array
        """
        return array * scale
    
    def save_models(self, suffix: str = "") -> None:
        """
        Save quantized models.
        
        Args:
            suffix: String suffix to add to filenames
        """
        if self.config.use_micrograd:
            # Save quantized micro-gradient models
            np.savez(f"{self.config.model_dir}/actor_quant_{suffix}.npz",
                     w1=self.actor_micro.w1 / self.actor_scales['w1'],
                     b1=self.actor_micro.b1 / self.actor_scales['b1'],
                     w2=self.actor_micro.w2 / self.actor_scales['w2'],
                     b2=self.actor_micro.b2 / self.actor_scales['b2'],
                     w3=self.actor_micro.w3 / self.actor_scales['w3'],
                     b3=self.actor_micro.b3 / self.actor_scales['b3'],
                     scales=np.array([
                         self.actor_scales['w1'],
                         self.actor_scales['b1'],
                         self.actor_scales['w2'],
                         self.actor_scales['b2'],
                         self.actor_scales['w3'],
                         self.actor_scales['b3']
                     ]))
            
            np.savez(f"{self.config.model_dir}/critic_quant_{suffix}.npz",
                     w1=self.critic_micro.w1 / self.critic_scales['w1'],
                     b1=self.critic_micro.b1 / self.critic_scales['b1'],
                     w2=self.critic_micro.w2 / self.critic_scales['w2'],
                     b2=self.critic_micro.b2 / self.critic_scales['b2'],
                     w3=self.critic_micro.w3 / self.critic_scales['w3'],
                     b3=self.critic_micro.b3 / self.critic_scales['b3'],
                     scales=np.array([
                         self.critic_scales['w1'],
                         self.critic_scales['b1'],
                         self.critic_scales['w2'],
                         self.critic_scales['b2'],
                         self.critic_scales['w3'],
                         self.critic_scales['b3']
                     ]))
        else:
            # For TensorFlow models, use TensorFlow Lite conversion
            super().save_models(suffix)
            
            # Convert to TF Lite with quantization
            converter = tf.lite.TFLiteConverter.from_saved_model(f"{self.config.model_dir}/actor_{suffix}")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            quantized_actor = converter.convert()
            
            converter = tf.lite.TFLiteConverter.from_saved_model(f"{self.config.model_dir}/critic_{suffix}")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            quantized_critic = converter.convert()
            
            # Save quantized models
            with open(f"{self.config.model_dir}/actor_quant_{suffix}.tflite", 'wb') as f:
                f.write(quantized_actor)
                
            with open(f"{self.config.model_dir}/critic_quant_{suffix}.tflite", 'wb') as f:
                f.write(quantized_critic)
        
        logger.info(f"Quantized models saved with suffix '{suffix}'")
    
    def load_models(self, suffix: str = "") -> None:
        """
        Load quantized models.
        
        Args:
            suffix: String suffix of the models to load
        """
        if self.config.use_micrograd:
            # Load quantized micro-gradient models
            actor_data = np.load(f"{self.config.model_dir}/actor_quant_{suffix}.npz")
            critic_data = np.load(f"{self.config.model_dir}/critic_quant_{suffix}.npz")
            
            # Extract scales
            actor_scales = actor_data['scales']
            critic_scales = critic_data['scales']
            
            # Update scales
            self.actor_scales = {
                'w1': actor_scales[0],
                'b1': actor_scales[1],
                'w2': actor_scales[2],
                'b2': actor_scales[3],
                'w3': actor_scales[4],
                'b3': actor_scales[5]
            }
            
            self.critic_scales = {
                'w1': critic_scales[0],
                'b1': critic_scales[1],
                'w2': critic_scales[2],
                'b2': critic_scales[3],
                'w3': critic_scales[4],
                'b3': critic_scales[5]
            }
            
            # Load quantized values
            self.actor_micro.w1 = actor_data['w1'] * self.actor_scales['w1']
            self.actor_micro.b1 = actor_data['b1'] * self.actor_scales['b1']
            self.actor_micro.w2 = actor_data['w2'] * self.actor_scales['w2']
            self.actor_micro.b2 = actor_data['b2'] * self.actor_scales['b2']
            self.actor_micro.w3 = actor_data['w3'] * self.actor_scales['w3']
            self.actor_micro.b3 = actor_data['b3'] * self.actor_scales['b3']
            
            self.critic_micro.w1 = critic_data['w1'] * self.critic_scales['w1']
            self.critic_micro.b1 = critic_data['b1'] * self.critic_scales['b1']
            self.critic_micro.w2 = critic_data['w2'] * self.critic_scales['w2']
            self.critic_micro.b2 = critic_data['b2'] * self.critic_scales['b2']
            self.critic_micro.w3 = critic_data['w3'] * self.critic_scales['w3']
            self.critic_micro.b3 = critic_data['b3'] * self.critic_scales['b3']
        else:
            # Load TF Lite models
            interpreter_actor = tf.lite.Interpreter(
                model_path=f"{self.config.model_dir}/actor_quant_{suffix}.tflite")
            interpreter_actor.allocate_tensors()
            
            interpreter_critic = tf.lite.Interpreter(
                model_path=f"{self.config.model_dir}/critic_quant_{suffix}.tflite")
            interpreter_critic.allocate_tensors()
            
            # Store interpreters
            self.actor_interpreter = interpreter_actor
            self.critic_interpreter = interpreter_critic
            
            # Get input and output details
            self.actor_input_index = interpreter_actor.get_input_details()[0]["index"]
            self.actor_output_indices = [
                detail["index"] for detail in interpreter_actor.get_output_details()
            ]
            
            self.critic_input_index = interpreter_critic.get_input_details()[0]["index"]
            self.critic_output_index = interpreter_critic.get_output_details()[0]["index"]
        
        logger.info(f"Quantized models loaded with suffix '{suffix}'")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Get action from quantized policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action
            
        Returns:
            Action and log probability
        """
        if self.config.use_micrograd:
            # Use regular micro-gradient forward pass (already quantized)
            return super().get_action(state, deterministic)
        else:
            # Use TF Lite interpreter
            state = np.array(state, dtype=np.float32)
            if len(state.shape) == 1:
                state = np.expand_dims(state, axis=0)
                
            self.actor_interpreter.set_tensor(self.actor_input_index, state)
            self.actor_interpreter.invoke()
            
            action_mean = self.actor_interpreter.get_tensor(self.actor_output_indices[0])[0]
            action_std = self.actor_interpreter.get_tensor(self.actor_output_indices[1])[0]
            
            if deterministic:
                return action_mean, 0.0
            else:
                # Sample from normal distribution and clip to [0, 1]
                action = np.clip(
                    np.random.normal(action_mean, action_std),
                    0, 1
                )
                
                # Compute log probability
                log_prob = -0.5 * np.sum(
                    np.square((action - action_mean) / action_std) + 
                    2 * np.log(action_std) + 
                    np.log(2 * np.pi)
                )
                
                return action, log_prob
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate from quantized critic.
        
        Args:
            state: Current state
            
        Returns:
            Value estimate
        """
        if self.config.use_micrograd:
            # Use regular micro-gradient forward pass (already quantized)
            return super().get_value(state)
        else:
            # Use TF Lite interpreter
            state = np.array(state, dtype=np.float32)
            if len(state.shape) == 1:
                state = np.expand_dims(state, axis=0)
                
            self.critic_interpreter.set_tensor(self.critic_input_index, state)
            self.critic_interpreter.invoke()
            
            value = self.critic_interpreter.get_tensor(self.critic_output_index)[0][0]
            
            return float(value)
        # Log probability calculation (simplified)
        log_prob = -np.sum(np.square(action - action_mean) / (2 * std**2))
        
        # PPO loss
        ratio = np.exp(log_prob - old_log_prob)
        
        # Clipped surrogate objective
        surrogate1 = ratio * advantage
        surrogate2 = np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
        loss = -min(surrogate1, surrogate2)
        
        # Compute gradients using chain rule
        # Output layer gradients
        dout = -(action - action_mean) / (std**2) * min(1.0, (1.0 + clip_ratio) / ratio if ratio > 1.0 else ratio / (1.0 - clip_ratio) if ratio < 1.0 else 1.0) * advantage
        dout_pre = dout * action_mean * (1 - action_mean)  # Sigmoid derivative
        
        # Second hidden layer gradients
        dw3 = np.outer(self.h2, dout_pre)
        db3 = dout_pre
        dh2 = np.dot(dout_pre, self.w3.T)
        dh2_pre = dh2 * (self.h2_pre > 0)  # ReLU derivative
        
        # First hidden layer gradients
        dw2 = np.outer(self.h1, dh2_pre)
        db2 = dh2_pre
        dh1 = np.dot(dh2_pre, self.w2.T)
        dh1_pre = dh1 * (self.h1_pre > 0)  # ReLU derivative
        
        # Input layer gradients
        dw1 = np.outer(x, dh1_pre)
        db1 = dh1_pre
        
        # Accumulate gradients
        self.dw1 += dw1
        self.db1 += db1
        self.dw2 += dw2
        self.db2 += db2
        self.dw3 += dw3
        self.db3 += db3
        
        return float(loss)
    
    def apply_gradients(self, batch_size: int) -> None:
        """
        Apply accumulated gradients with SGD.
        
        Args:
            batch_size: Number of samples in the batch
        """
        # Normalize by batch size
        self.dw1 /= batch_size
        self.db1 /= batch_size
        self.dw2 /= batch_size
        self.db2 /= batch_size
        self.dw3 /= batch_size
        self.db3 /= batch_size
        
        # Update weights with SGD
        self.w1 -= self.learning_rate * self.dw1
        self.b1 -= self.learning_rate * self.db1
        self.w2 -= self.learning_rate * self.dw2
        self.b2 -= self.learning_rate * self.db2
        self.w3 -= self.learning_rate * self.dw3
        self.b3 -= self.learning_rate * self.db3
        
        # Reset gradients
        self.dw1.fill(0)
        self.db1.fill(0)
        self.dw2.fill(0)
        self.db2.fill(0)
        self.dw3.fill(0)
        self.db3.fill(0)
    
    def save(self, filepath: str) -> None:
        """
        Save model weights to file.
        
        Args:
            filepath: Path to save weights
        """
        np.savez(filepath, 
                 w1=self.w1, b1=self.b1, 
                 w2=self.w2, b2=self.b2, 
                 w3=self.w3, b3=self.b3)
    
    def load(self, filepath: str) -> None:
        """
        Load model weights from file.
        
        Args:
            filepath: Path to load weights from
        """
        weights = np.load(filepath)
        self.w1 = weights['w1']
        self.b1 = weights['b1']
        self.w2 = weights['w2']
        self.b2 = weights['b2']
        self.w3 = weights['w3']
        self.b3 = weights['b3']


class MicroGradCritic:
    """
    Lightweight neural network for value function using micro-gradient framework.
    
    This implementation uses a custom micro-gradient framework for minimal memory footprint,
    suitable for embedded applications or when memory is constrained.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 64, learning_rate: float = 0.001):
        """
        Initialize micro-gradient critic network.
        
        Args:
            state_dim: Dimension of the state space
            hidden_dim: Size of hidden layers
            learning_rate: Learning rate for SGD
        """
        # Initialize weights with Xavier/Glorot initialization
        scale1 = np.sqrt(2.0 / (state_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        scale3 = np.sqrt(2.0 / (hidden_dim + 1))
        
        # Define network architecture
        self.w1 = np.random.randn(state_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * scale2
        self.b2 = np.zeros(hidden_dim)
        self.w3 = np.random.randn(hidden_dim, 1) * scale3
        self.b3 = np.zeros(1)
        
        # Training parameters
        self.learning_rate = learning_rate
        
        # Gradients
        self.dw1 = np.zeros_like(self.w1)
        self.db1 = np.zeros_like(self.b1)
        self.dw2 = np.zeros_like(self.w2)
        self.db2 = np.zeros_like(self.b2)
        self.dw3 = np.zeros_like(self.w3)
        self.db3 = np.zeros_like(self.b3)
        
    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass through the network.
        
        Args:
            x: Input state
            
        Returns:
            Value estimate
        """
        # First layer with ReLU activation
        self.h1_pre = np.dot(x, self.w1) + self.b1
        self.h1 = np.maximum(0, self.h1_pre)  # ReLU
        
        # Second layer with ReLU activation
        self.h2_pre = np.dot(self.h1, self.w2) + self.b2
        self.h2 = np.maximum(0, self.h2_pre)  # ReLU
        
        # Output layer (linear activation for value)
        self.out = np.dot(self.h2, self.w3) + self.b3
        
        return float(self.out[0])
    
    def backward(self, x: np.ndarray, target_value: float) -> float:
        """
        Backward pass with MSE loss.
        
        Args:
            x: Input state
            target_value: Target value (return)
            
        Returns:
            Loss value
        """
        # Forward pass
        value = self.forward(x)
        
        # MSE loss
        loss = 0.5 * (value - target_value)**2
        
        # Gradients
        # Output layer gradients
        dout = value - target_value
        
        # Second hidden layer gradients
        dw3 = np.outer(self.h2, dout)
        db3 = dout
        dh2 = dout * self.w3.T
        dh2_pre = dh2 * (self.h2_pre > 0)  # ReLU derivative
        
        # First hidden layer gradients
        dw2 = np.outer(self.h1, dh2_pre)
        db2 = dh2_pre
        dh1 = np.dot(dh2_pre, self.w2.T)
        dh1_pre = dh1 * (self.h1_pre > 0)  # ReLU derivative
        
        # Input layer gradients
        dw1 = np.outer(x, dh1_pre)
        db1 = dh1_pre
        
        # Accumulate gradients
        self.dw1 += dw1
        self.db1 += db1
        self.dw2 += dw2
        self.db2 += db2
        self.dw3 += dw3
        self.db3 += db3
        
        return float(loss)
    
    def apply_gradients(self, batch_size: int) -> None:
        """
        Apply accumulated gradients with SGD.
        
        Args:
            batch_size: Number of samples in the batch
        """
        # Normalize by batch size
        self.dw1 /= batch_size
        self.db1 /= batch_size
        self.dw2 /= batch_size
        self.db2 /= batch_size
        self.dw3 /= batch_size
        self.db3 /= batch_size
        
        # Update weights with SGD
        self.w1 -= self.learning_rate * self.dw1
        self.b1 -= self.learning_rate * self.db1
        self.w2 -= self.learning_rate * self.dw2
        self.b2 -= self.learning_rate * self.db2
        self.w3 -= self.learning_rate * self.dw3
        self.b3 -= self.learning_rate * self.db3
        
        # Reset gradients
        self.dw1.fill(0)
        self.db1.fill(0)
        self.dw2.fill(0)
        self.db2.fill(0)
        self.dw3.fill(0)
        self.db3.fill(0)
    
    def save(self, filepath: str) -> None:
        """
        Save model weights to file.
        
        Args:
            filepath: Path to save weights
        """
        np.savez(filepath, 
                 w1=self.w1, b1=self.b1, 
                 w2=self.w2, b2=self.b2, 
                 w3=self.w3, b3=self.b3)
    
    def load(self, filepath: str) -> None:
        """
        Load model weights from file.
        
        Args:
            filepath: Path to load weights from
        """
        weights = np.load(filepath)
        self.w1 = weights['w1']
        self.b1 = weights['b1']
        self.w2 = weights['w2']
        self.b2 = weights['b2']
        self.w3 = weights['w3']
        self.b3 = weights['b3']


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for learning optimal magnetic field configurations.
    
    This agent uses either TensorFlow or a lightweight custom micro-gradient framework
    for training, with optimizations for memory efficiency.
    """
    
    def __init__(self, config: PPOAgentConfig):
        """
        Initialize the PPO agent.
        
        Args:
            config: Configuration object for the agent
        """
        self.config = config
        self.config.validate()
        
        # Create model directory if it doesn't exist
        if not os.path.exists(config.model_dir):
            os.makedirs(config.model_dir)
        
        # Use either micro-gradient framework or TensorFlow
        if config.use_micrograd:
            # Initialize micro-gradient models
            self.actor_micro = MicroGradActor(
                config.state_dim, 
                config.action_dim, 
                config.hidden_dim, 
                config.lr_actor
            )
            
            self.critic_micro = MicroGradCritic(
                config.state_dim, 
                config.hidden_dim, 
                config.lr_critic
            )
            
            # Set TensorFlow models to None
            self.actor = None
            self.critic = None
        else:
            # Build TensorFlow models
            self.actor = self._build_actor_network(config.hidden_dim)
            self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=config.lr_actor))
            
            self.critic = self._build_critic_network(config.hidden_dim)
            self.critic.compile(optimizer=keras.optimizers.Adam(learning_rate=config.lr_critic))
            
            # Set micro-gradient models to None
            self.actor_micro = None
            self.critic_micro = None
        
        # Create memory buffer
        self.memory = PPOMemory()
        
        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_values = []
        
        logger.info(f"PPO Agent initialized with state_dim={config.state_dim}, action_dim={config.action_dim}")
    
    def _build_actor_network(self, hidden_dim: int) -> keras.Model:
        """
        Build the actor network (policy) using TensorFlow.
        
        Args:
            hidden_dim: Size of hidden layers
            
        Returns:
            Actor network model
        """
        # Input is the state
        state_input = layers.Input(shape=(self.config.state_dim,))
        
        # Hidden layers with Layer Normalization for stable training
        x = layers.Dense(hidden_dim)(state_input)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Dense(hidden_dim)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Output are the magnetic field strengths (between 0 and 1)
        mean = layers.Dense(self.config.action_dim, activation='sigmoid')(x)
        
        # Fixed standard deviation for exploration
        log_std = layers.Lambda(lambda x: tf.fill(tf.shape(x), np.log(0.1)))(mean)
        std = layers.Lambda(lambda x: tf.exp(x))(log_std)
        
        # Create the model
        model = keras.Model(inputs=state_input, outputs=[mean, std])
        return model
    
    def _build_critic_network(self, hidden_dim: int) -> keras.Model:
        """
        Build the critic network (value function) using TensorFlow.
        
        Args:
            hidden_dim: Size of hidden layers
            
        Returns:
            Critic network model
        """
        # Input is the state
        state_input = layers.Input(shape=(self.config.state_dim,))
        
        # Hidden layers with Layer Normalization
        x = layers.Dense(hidden_dim)(state_input)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Dense(hidden_dim)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Output is the value estimate
        value = layers.Dense(1)(x)
        
        # Create the model
        model = keras.Model(inputs=state_input, outputs=value)
        return model
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Get an action from the policy for the given state.
        
        Args:
            state: Current state observation
            deterministic: Whether to return deterministic actions
            
        Returns:
            Action and corresponding action log probability
        """
        # Ensure state is correctly shaped
        state = np.array(state, dtype=np.float32)
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        
        if self.config.use_micrograd:
            # Use micro-gradient actor
            action_mean = self.actor_micro.forward(state[0])
            
            if deterministic:
                # Return deterministic action
                return action_mean, 0.0
            else:
                # Add some noise for exploration
                noise_scale = 0.1
                action = np.clip(
                    action_mean + noise_scale * np.random.randn(self.config.action_dim),
                    0, 1
                )
                
                # Compute log probability (approximate)
                log_prob = -np.sum(np.square(action - action_mean) / (2 * noise_scale**2))
                
                return action, log_prob
        else:
            # Use TensorFlow actor
            action_mean, action_std = self.actor(state)
            action_mean = action_mean[0].numpy()
            action_std = action_std[0].numpy()
            
            if deterministic:
                # Return deterministic action
                return action_mean, 0.0
            else:
                # Sample from normal distribution and clip to [0, 1]
                action = np.clip(
                    np.random.normal(action_mean, action_std),
                    0, 1
                )
                
                # Compute log probability
                log_prob = -0.5 * np.sum(
                    np.square((action - action_mean) / action_std) + 
                    2 * np.log(action_std) + 
                    np.log(2 * np.pi)
                )
                
                return action, log_prob
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate for the given state.
        
        Args:
            state: Current state observation
            
        Returns:
            Value estimate
        """
        # Ensure state is correctly shaped
        state = np.array(state, dtype=np.float32)
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        
        if self.config.use_micrograd:
            # Use micro-gradient critic
            return self.critic_micro.forward(state[0])
        else:
            # Use TensorFlow critic
            return float(self.critic(state)[0, 0].numpy())
    
    def update(self, states: np.ndarray, actions: np.ndarray, old_log_probs: np.ndarray,
             returns: np.ndarray, advantages: np.ndarray, batch_size: int, epochs: int) -> Dict[str, float]:
        """
        Update policy and value function using PPO.
        
        Args:
            states: Array of observed states
            actions: Array of actions taken
            old_log_probs: Array of log probabilities of actions under old policy
            returns: Array of computed returns
            advantages: Array of computed advantages
            batch_size: Batch size for training
            epochs: Number of epochs per update
            
        Returns:
            Dictionary of training metrics
        """
        # Get dataset size
        dataset_size = len(states)
        
        # Training metrics
        actor_losses = []
        critic_losses = []
        entropy_values = []
        
        if self.config.use_micrograd:
            # Update using micro-gradient framework
            for epoch in range(epochs):
                # Generate random indices for batching
                indices = np.random.permutation(dataset_size)
                
                # Process in batches
                epoch_actor_losses = []
                epoch_critic_losses = []
                
                for start_idx in range(0, dataset_size, batch_size):
                    # Get batch indices
                    batch_indices = indices[start_idx:min(start_idx + batch_size, dataset_size)]
                    actual_batch_size = len(batch_indices)
                    
                    # Update actor
                    batch_actor_losses = []
                    for i in batch_indices:
                        loss = self.actor_micro.backward(
                            states[i],
                            actions[i],
                            float(advantages[i]),
                            float(old_log_probs[i]),
                            self.config.clip_ratio
                        )
                        batch_actor_losses.append(loss)
                    
                    self.actor_micro.apply_gradients(actual_batch_size)
                    epoch_actor_losses.extend(batch_actor_losses)
                    
                    # Update critic
                    batch_critic_losses = []
                    for i in batch_indices:
                        loss = self.critic_micro.backward(
                            states[i],
                            float(returns[i])
                        )
                        batch_critic_losses.append(loss)
                    
                    self.critic_micro.apply_gradients(actual_batch_size)
                    epoch_critic_losses.extend(batch_critic_losses)
                
                # Record metrics
                actor_losses.append(np.mean(epoch_actor_losses))
                critic_losses.append(np.mean(epoch_critic_losses))
                # Entropy is fixed in micro-gradient framework
                entropy_values.append(0.0)
        else:
            # Update using TensorFlow
            for epoch in range(epochs):
                # Generate random indices for batching
                indices = np.random.permutation(dataset_size)
                
                # Process in batches
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_entropy_values = []
                
                for start_idx in range(0, dataset_size, batch_size):
                    # Get batch indices
                    batch_indices = indices[start_idx:min(start_idx + batch_size, dataset_size)]
                    
                    # Extract batch data
                    batch_states = tf.convert_to_tensor(states[batch_indices], dtype=tf.float32)
                    batch_actions = tf.convert_to_tensor(actions[batch_indices], dtype=tf.float32)
                    batch_old_log_probs = tf.convert_to_tensor(old_log_probs[batch_indices], dtype=tf.float32)
                    batch_returns = tf.convert_to_tensor(returns[batch_indices], dtype=tf.float32)
                    batch_advantages = tf.convert_to_tensor(advantages[batch_indices], dtype=tf.float32)
                    
                    # Update critic
                    with tf.GradientTape() as tape:
                        # Current value predictions
                        values_pred = self.critic(batch_states)
                        # MSE loss
                        critic_loss = tf.reduce_mean(tf.square(batch_returns - values_pred))
                    
                    # Get gradients and update critic
                    critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                    self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                    
                    # Update actor
                    with tf.GradientTape() as tape:
                        # Get current action distribution
                        action_mean, action_std = self.actor(batch_states)
                        
                        # Compute current log probabilities
                        current_log_probs = -0.5 * tf.reduce_sum(
                            tf.square((batch_actions - action_mean) / action_std) +
                            2 * tf.math.log(action_std) +
                            tf.math.log(2 * np.pi),
                            axis=1
                        )
                        
                        # Compute ratio
                        ratio = tf.exp(current_log_probs - batch_old_log_probs)
                        
                        # Compute surrogate losses
                        surrogate1 = ratio * batch_advantages
                        surrogate2 = tf.clip_by_value(
                            ratio, 
                            1.0 - self.config.clip_ratio, 
                            1.0 + self.config.clip_ratio
                        ) * batch_advantages
                        
                        # Compute actor loss (negative because we want to maximize the objective)
                        actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                        
                        # Add entropy bonus for exploration
                        entropy = tf.reduce_mean(
                            0.5 * tf.math.log(2 * np.pi * tf.square(action_std)) + 0.5
                        )
                        actor_loss -= self.config.entropy_coef * entropy
                    
                    # Get gradients and update actor
                    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                    self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                    
                    # Record metrics
                    epoch_actor_losses.append(actor_loss.numpy())
                    epoch_critic_losses.append(critic_loss.numpy())
                    epoch_entropy_values.append(entropy.numpy())
                
                # Record metrics
                actor_losses.append(np.mean(epoch_actor_losses))
                critic_losses.append(np.mean(epoch_critic_losses))
                entropy_values.append(np.mean(epoch_entropy_values))
        
        # Update metrics
        self.actor_losses.extend(actor_losses)
        self.critic_losses.extend(critic_losses)
        self.entropy_values.extend(entropy_values)
        
        # Return training metrics
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropy_values)
        }
    
    def learn(self, env, max_episodes: int = 1000, max_steps: int = 200, 
            update_freq: int = 2048, batch_size: int = 64, epochs: int = 10,
            save_freq: int = 100, gamma: float = 0.99, gae_lambda: float = 0.95,
            render: bool = False) -> Dict[str, List[float]]:
        """
        Learn policy from environment interactions.
        
        Args:
            env: Environment to learn from
            max_episodes: Maximum number of episodes
            max_steps: Maximum steps per episode
            update_freq: Frequency of policy updates
            batch_size: Batch size for training
            epochs: Number of epochs per update
            save_freq: Frequency of model saving
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            render: Whether to render environment
            
        Returns:
            Dictionary of training metrics
        """
        # Initialize metrics
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'thrust_values': [],
            'efficiency_values': [],
            'escape_rates': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropy_values': []
        }
        
        # Initialize step counter
        total_steps = 0
        
        # Training loop
        for episode in range(max_episodes):
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            
            # Episode loop
            for step in range(max_steps):
                # Get action and value estimate
                action, log_prob = self.get_action(state)
                value = self.get_value(state)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store experience in memory
                self.memory.add(state, action, log_prob, reward, value, done)
                
                # Update state and accumulate reward
                state = next_state
                episode_reward += reward
                
                # Update total steps
                total_steps += 1
                
                # Update policy if enough steps collected
                if total_steps % update_freq == 0:
                    # Compute last value for bootstrapping
                    last_value = self.get_value(state)
                    
                    # Compute advantages and returns
                    self.memory.compute_advantages(gamma, gae_lambda, last_value, done)
                    
                    # Get experiences
                    experiences = self.memory.get()
                    
                    # Update policy and value function
                    update_metrics = self.update(
                        experiences['states'],
                        experiences['actions'],
                        experiences['log_probs'],
                        experiences['returns'],
                        experiences['advantages'],
                        batch_size,
                        epochs
                    )
                    
                    # Record metrics
                    metrics['actor_losses'].append(update_metrics['actor_loss'])
                    metrics['critic_losses'].append(update_metrics['critic_loss'])
                    metrics['entropy_values'].append(update_metrics['entropy'])
                    
                    # Clear memory
                    self.memory.clear()
                
                # Render if enabled
                if render:
                    env.render()
                
                # End episode if done
                if done:
                    break
            
            # Store episode metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(step + 1)
            metrics['thrust_values'].append(info.get('thrust', 0))
            metrics['efficiency_values'].append(info.get('efficiency', 0))
            metrics['escape_rates'].append(info.get('escape_rate', 0))
            
            # Print episode statistics
            logger.info(f"Episode {episode}: Reward={episode_reward:.3f}, "
                       f"Steps={step+1}, Thrust={info.get('thrust', 0):.3e}, "
                       f"Efficiency={info.get('efficiency', 0):.3e}, "
                       f"Escape Rate={info.get('escape_rate', 0):.3f}")
            
            # Save models periodically
            if (episode + 1) % save_freq == 0:
                self.save_models(f"{episode+1}")
        
        # Final save
        self.save_models("final")
        
        return metrics
    
    def save_models(self, suffix: str = "") -> None:
        """
        Save actor and critic models.
        
        Args:
            suffix: String suffix to add to filenames
        """
        if self.config.use_micrograd:
            # Save micro-gradient models
            self.actor_micro.save(f"{self.config.model_dir}/actor_micro_{suffix}.npz")
            self.critic_micro.save(f"{self.config.model_dir}/critic_micro_{suffix}.npz")
        else:
            # Save TensorFlow models
            self.actor.save(f"{self.config.model_dir}/actor_{suffix}")
            self.critic.save(f"{self.config.model_dir}/critic_{suffix}")
        
        logger.info(f"Models saved with suffix '{suffix}'")
    
    def load_models(self, suffix: str = "") -> None:
        """
        Load actor and critic models.
        
        Args:
            suffix: String suffix of the models to load
        """
        if self.config.use_micrograd:
            # Load micro-gradient models
            self.actor_micro.load(f"{self.config.model_dir}/actor_micro_{suffix}.npz")
            self.critic_micro.load(f"{self.config.model_dir}/critic_micro_{suffix}.npz")
        else:
            # Load TensorFlow models
            self.actor = keras.models.load_model(f"{self.config."""
Reinforcement learning agent for FFRE magnetic field optimization.

This module provides a PPO-based agent that learns to optimize magnetic field
configurations for maximum thrust in a Fission Fragment Rocket Engine.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import logging
from typing import Tuple, List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
from collections import deque
import time

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PPOAgentConfig:
    """Configuration for PPO agent."""
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    lr_actor: float = 0.0003
    lr_critic: float = 0.001
    gamma: float = 0.99
    clip_ratio: float = 0.2
    critic_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    model_dir: str = "./models"
    use_micrograd: bool = False  # Use custom micro-gradient framework for low memory footprint
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.state_dim > 0, "State dimension must be positive"
        assert self.action_dim > 0, "Action dimension must be positive"
        assert self.hidden_dim > 0, "Hidden dimension must be positive"
        assert self.lr_actor > 0, "Actor learning rate must be positive"
        assert self.lr_critic > 0, "Critic learning rate must be positive"
        assert 0 < self.gamma < 1, "Gamma must be between 0 and 1"
        assert self.clip_ratio > 0, "Clip ratio must be positive"


class PPOMemory:
    """
    Memory buffer for PPO algorithm with efficient storage.
    
    This class manages experience collection and batch sampling for PPO,
    optimized for minimal memory overhead.
    """
    
    def __init__(self, batch_size: int = 64):
        """
        Initialize memory buffer.
        
        Args:
            batch_size: Batch size for training
        """
        self.batch_size = batch_size
        self.clear()
        
    def clear(self) -> None:
        """Clear memory buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
    def add(self, state: np.ndarray, action: np.ndarray, log_prob: float,
          reward: float, value: float, done: bool) -> None:
        """
        Add an experience to memory.
        
        Args:
            state: Observation
            action: Action taken
            log_prob: Log probability of action
            reward: Reward received
            value: Value estimate
            done: Whether episode terminated
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
    def compute_advantages(self, gamma: float, lam: float, last_value: float, last_done: bool) -> None:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            gamma: Discount factor
            lam: GAE lambda parameter
            last_value: Value estimate of last state
            last_done: Whether last state was terminal
        """
        # Convert to numpy arrays for vectorized operations
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [last_done])
        
        # Compute returns and advantages using GAE
        gae = 0
        self.returns = []
        self.advantages = []
        
        # Iterate in reverse for efficient calculation
        for step in reversed(range(len(rewards))):
            # Calculate delta = r + gamma * V(s') * (1 - done) - V(s)
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            
            # Update GAE
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            
            # Store return and advantage
            self.returns.insert(0, gae + values[step])
            self.advantages.insert(0, gae)
        
        # Convert to numpy arrays
        self.returns = np.array(self.returns, dtype=np.float32)
        self.advantages = np.array(self.advantages, dtype=np.float32)
        
        # Normalize advantages for stable training
        self.advantages = (self.advantages - np.mean(self.advantages)) / (np.std(self.advantages) + 1e-8)
        
    def get(self) -> Dict[str, np.ndarray]:
        """
        Get all stored experiences.
        
        Returns:
            Dictionary of experience tensors
        """
        return {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.float32),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.float32),
            'advantages': self.advantages,
            'returns': self.returns
        }
        
    def get_batch_iterator(self) -> Dict[str, np.ndarray]:
        """
        Generator for batches of experiences.
        
        Yields:
            Dictionary of batch tensors
        """
        # Get buffer size
        buffer_size = len(self.states)
        
        # Get indices
        indices = np.random.permutation(buffer_size)
        
        # Yield batches
        for start_idx in range(0, buffer_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, buffer_size)
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                'states': np.array([self.states[i] for i in batch_indices], dtype=np.float32),
                'actions': np.array([self.actions[i] for i in batch_indices], dtype=np.float32),
                'log_probs': np.array([self.log_probs[i] for i in batch_indices], dtype=np.float32),
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices]
            }
    
    def size(self) -> int:
        """
        Get number of experiences in memory.
        
        Returns:
            Number of experiences
        """
        return len(self.states)


class MicroGradActor:
    """
    Lightweight neural network for policy function using micro-gradient framework.
    
    This implementation uses a custom micro-gradient framework for minimal memory footprint,
    suitable for embedded applications or when memory is constrained.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, learning_rate: float = 0.001):
        """
        Initialize micro-gradient actor network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Size of hidden layers
            learning_rate: Learning rate for SGD
        """
        # Initialize weights with Xavier/Glorot initialization
        scale1 = np.sqrt(2.0 / (state_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        scale3 = np.sqrt(2.0 / (hidden_dim + action_dim))
        
        # Define network architecture
        self.w1 = np.random.randn(state_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * scale2
        self.b2 = np.zeros(hidden_dim)
        self.w3 = np.random.randn(hidden_dim, action_dim) * scale3
        self.b3 = np.zeros(action_dim)
        
        # Training parameters
        self.learning_rate = learning_rate
        
        # Gradients
        self.dw1 = np.zeros_like(self.w1)
        self.db1 = np.zeros_like(self.b1)
        self.dw2 = np.zeros_like(self.w2)
        self.db2 = np.zeros_like(self.b2)
        self.dw3 = np.zeros_like(self.w3)
        self.db3 = np.zeros_like(self.b3)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input state
            
        Returns:
            Action means (between 0 and 1)
        """
        # First layer with ReLU activation
        self.h1_pre = np.dot(x, self.w1) + self.b1
        self.h1 = np.maximum(0, self.h1_pre)  # ReLU
        
        # Second layer with ReLU activation
        self.h2_pre = np.dot(self.h1, self.w2) + self.b2
        self.h2 = np.maximum(0, self.h2_pre)  # ReLU
        
        # Output layer with sigmoid activation for [0, 1] range
        self.out_pre = np.dot(self.h2, self.w3) + self.b3
        self.out = 1.0 / (1.0 + np.exp(-self.out_pre))  # Sigmoid
        
        return self.out
    
    def backward(self, x: np.ndarray, action: np.ndarray, advantage: float, old_log_prob: float, clip_ratio: float) -> float:
        """
        Backward pass with PPO loss.
        
        Args:
            x: Input state
            action: Action taken
            advantage: Advantage estimate
            old_log_prob: Log probability from old policy
            clip_ratio: PPO clipping parameter
            
        Returns:
            Loss value
        """
        # Forward pass
        action_mean = self.forward(x)
        
        # Assume fixed standard deviation for simplicity
        std = 0.1
        
        # Log probability calculation (simplified)
        log_prob = -np.sum(np.square(action - action_mean) / (2 * std**2))
        
        # PPO loss
        ratio = np.exp(log_prob - ol