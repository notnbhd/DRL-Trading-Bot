import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay
import tensorflow.keras.backend as K
from src.models.cnn_lstm_model import CNNLSTM

class PPOAgent:
    """PPO Agent for cryptocurrency trading"""
    
    def __init__(
        self,
        input_shape,
        action_space,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        lam=0.95,
        kl_target=0.02,  # Target KL divergence
        kl_cutoff_factor=3.0,  # KL cutoff factor
        adaptive_epsilon=True,  # Whether to use adaptive epsilon
        use_lr_schedule=False  # Whether to use learning rate scheduling
    ):
        """
        Initialize the PPO agent
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of the input data (lookback_window, features)
        action_space : int
            Number of possible actions
        learning_rate : float, optional
            Learning rate for the Adam optimizer
        gamma : float, optional
            Discount factor for future rewards
        epsilon : float, optional
            Clipping parameter for PPO
        value_coef : float, optional
            Coefficient for value loss
        entropy_coef : float, optional
            Coefficient for entropy loss
        lam : float, optional
            GAE parameter for advantage estimation
        kl_target : float, optional
            Target KL divergence
        kl_cutoff_factor : float, optional
            Factor to determine KL cutoff
        adaptive_epsilon : bool, optional
            Whether to use adaptive epsilon based on KL divergence
        use_lr_schedule : bool, optional
            Whether to use learning rate scheduling
        """
        self.input_shape = input_shape
        self.action_space = action_space
        self.initial_learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon  # Used in the clipping function
        self.initial_epsilon = epsilon  # Store initial epsilon for possible reset
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.lam = lam
        self.use_lr_schedule = use_lr_schedule
        self.training_steps = 0  # Track total training steps for scheduling
        
        # KL divergence parameters
        self.kl_target = kl_target
        self.kl_cutoff_factor = kl_cutoff_factor
        self.adaptive_epsilon = adaptive_epsilon
        
        # Create CNN-LSTM model
        self.model = CNNLSTM(input_shape, action_space, learning_rate)
        
        # Actor and critic models
        self.actor = self.model.get_actor()
        self.critic = self.model.get_critic()
        
        # Create optimizers with learning rate scheduling
        if use_lr_schedule:
            # Exponential decay learning rate schedule for actor
            actor_lr_schedule = ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=10000,  # Decay every 10k steps
                decay_rate=0.95,    # Reduce by 5% each decay step
                staircase=True      # Apply decay in discrete steps
            )
            
            # Piecewise constant decay for critic - starts higher, declines more aggressively 
            critic_lr_schedule = PiecewiseConstantDecay(
                boundaries=[5000, 15000, 30000],  # Step boundaries
                values=[learning_rate * 3.0, learning_rate * 2.0, learning_rate * 1.0, learning_rate * 0.5]  # Learning rates for each period
            )
            
            self.actor_optimizer = Adam(learning_rate=actor_lr_schedule)
            self.critic_optimizer = Adam(learning_rate=critic_lr_schedule)
            
            # Store schedules for debugging
            self.actor_lr_schedule = actor_lr_schedule
            self.critic_lr_schedule = critic_lr_schedule
        else:
            # Fixed learning rates
            self.actor_optimizer = Adam(learning_rate=0.0003)
            self.critic_optimizer = Adam(learning_rate=0.0005)
        
        # Initialize memory for trajectory collection
        self.clear_memory()
    
    def get_action(self, state, training=True, tau=0.1):
        """
        Get action from the actor model

        Parameters:
        -----------
        state : numpy.ndarray
            Current state observation
        training : bool, optional
            Whether in training mode (random sampling) or not (softmax sampling with temperature)
        tau : float, optional
            Temperature for softmax sampling in testing mode (lower = more deterministic)

        Returns:
        --------
        action, action_prob
        """
        # Reshape state if needed
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)

        # Get action probabilities from the policy network (π_θ(at|st))
        action_probs = self.actor.predict(state, verbose=0)[0]

        if training:
            # Sample action from probability distribution
            action = np.random.choice(self.action_space, p=action_probs)
        else:
            action = np.argmax(action_probs)

        return action, action_probs

    def remember(self, state, action, reward, next_state, done, action_probs):
        """
        Store experience in memory for trajectory collection
        """
        # Convert states to float32 to reduce memory usage
        self.states.append(state.astype(np.float32) if isinstance(state, np.ndarray) else state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state.astype(np.float32) if isinstance(next_state, np.ndarray) else next_state)
        self.dones.append(done)
        self.action_probs.append(action_probs)
    
    def _calculate_kl_divergence(self, old_probs, new_probs):
        """
        Calculate KL divergence between old and new policy distributions
        
        Parameters:
        -----------
        old_probs : tensorflow.Tensor
            Probabilities from old policy
        new_probs : tensorflow.Tensor
            Probabilities from new policy
            
        Returns:
        --------
        Mean KL divergence
        """
        # Ensure both tensors have the same dtype (float32)
        old_probs = tf.cast(old_probs, tf.float32)
        new_probs = tf.cast(new_probs, tf.float32)
        
        # Avoid log(0) by adding a small epsilon
        old_probs = tf.clip_by_value(old_probs, 1e-10, 1.0)
        new_probs = tf.clip_by_value(new_probs, 1e-10, 1.0)
        
        # Calculate KL divergence: KL(p||q) = sum(p * log(p/q))
        kl_div = tf.reduce_sum(
            old_probs * tf.math.log(old_probs / new_probs),
            axis=1
        )
        return tf.reduce_mean(kl_div)
    
    def _adjust_epsilon(self, kl_div):
        """
        Adjust epsilon based on KL divergence to prevent policy collapse
        
        Parameters:
        -----------
        kl_div : float
            KL divergence between old and new policies
            
        Returns:
        --------
        Updated epsilon value
        """
        # If KL divergence is too high, reduce epsilon to make updates more conservative
        if kl_div > self.kl_cutoff_factor * self.kl_target:
            return max(self.epsilon * 0.5, 0.01)  # Reduce epsilon but not below 0.01
        
        # If KL divergence is too low, increase epsilon to allow larger updates
        elif kl_div < self.kl_target / self.kl_cutoff_factor:
            return min(self.epsilon * 1.5, self.initial_epsilon)  # Increase epsilon but not above initial
        
        # Otherwise, keep epsilon the same
        return self.epsilon
    
    def _compute_advantage(self, rewards, values, next_values, dones):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        This computes Ât in the PPO formula: L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
        
        Parameters:
        -----------
        rewards : list
            List of rewards
        values : list
            List of state values from critic
        next_values : list
            List of next state values from critic
        dones : list
            List of done flags
            
        Returns:
        --------
        advantages, returns
        """
        # Initialize advantage array
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        
        # Compute advantages using GAE (backwards)
        for t in reversed(range(len(rewards))):
            # Get next value (0 if terminal state)
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            # TD error delta = reward + gamma * next_value * (1 - done) - value
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE formula: A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Compute returns (value targets)
        returns = advantages + values


        # Normalize advantages for training stability
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns
    
    def get_learning_rates(self):
        """Get current learning rates for actor and critic optimizers"""
        if self.use_lr_schedule:
            actor_lr = self.actor_lr_schedule(self.training_steps)
            critic_lr = self.critic_lr_schedule(self.training_steps)
        else:
            actor_lr = self.actor_optimizer.learning_rate
            critic_lr = self.critic_optimizer.learning_rate
            
        return {"actor_lr": float(actor_lr), "critic_lr": float(critic_lr)}
    
    def train(self, batch_size=32, epochs=5):
        """
        Train the PPO agent following the flowchart in Figure 4 and
        pseudocode in Figure 5
        
        Parameters:
        -----------
        batch_size : int, optional
            Size of mini-batches for training
        epochs : int, optional
            Number of epochs to train on the same data
            
        Returns:
        --------
        Dictionary with training metrics
        """
        # Check if we have enough data
        if len(self.states) < batch_size:
            return {'actor_loss': [], 'critic_loss': [], 'total_loss': [], 'kl_div': []}
        
        # Convert to numpy arrays with explicit float32 dtype to reduce memory usage
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards, dtype=np.float32)
        next_states = np.array(self.next_states, dtype=np.float32)
        dones = np.array(self.dones)
        old_action_probs = np.array(self.action_probs, dtype=np.float32)
        
        # Get values for current states and next states using critic
        # Move prediction to CPU to avoid GPU memory issues
        with tf.device('/GPU:0'):
            values = self.critic.predict(states, verbose=0).flatten()
            next_values = self.critic.predict(next_states, verbose=0).flatten()
        
        # Compute advantages and returns using GAE
        advantages, returns = self._compute_advantage(rewards, values, next_values, dones)
        
        # Create one-hot encoded actions
        actions_one_hot = tf.one_hot(actions, self.action_space)
        
        # Track training metrics
        history = {'actor_loss': [], 'critic_loss': [], 'total_loss': [], 'kl_div': []}
        
        # Implement PPO training loop with multiple epochs
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # Keep track of average KL divergence for this epoch
            epoch_kl_divs = []
            
            # Process mini-batches
            for start_idx in range(0, len(indices), batch_size):
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data and convert to tensors
                with tf.device('/GPU:0'):
                    batch_states = tf.convert_to_tensor(states[batch_indices], dtype=tf.float32)
                    batch_actions_one_hot = tf.gather(actions_one_hot, batch_indices)
                    batch_advantages = tf.convert_to_tensor(advantages[batch_indices], dtype=tf.float32)
                    batch_returns = tf.convert_to_tensor(returns[batch_indices], dtype=tf.float32)
                    batch_old_probs = tf.convert_to_tensor(old_action_probs[batch_indices], dtype=tf.float32)
                
                # Train actor
                with tf.GradientTape() as tape:
                    # Get current policy probabilities
                    current_probs = self.actor(batch_states, training=True)
                    
                    # Calculate KL divergence between old and new policies
                    kl_div = self._calculate_kl_divergence(batch_old_probs, current_probs)
                    epoch_kl_divs.append(float(kl_div))
                    
                    # Make sure both tensors have the same dtype before multiplication
                    current_probs_float32 = tf.cast(current_probs, tf.float32)
                    batch_actions_one_hot_float32 = tf.cast(batch_actions_one_hot, tf.float32)
                    
                    # Extract probabilities of the actions that were actually taken
                    current_action_probs = tf.reduce_sum(current_probs_float32 * batch_actions_one_hot_float32, axis=1)
                    old_action_prob_values = tf.reduce_sum(batch_old_probs * batch_actions_one_hot_float32, axis=1)
                    
                    # Calculate probability ratio
                    ratio = current_action_probs / (old_action_prob_values + 1e-8)
                    
                    # Adaptive epsilon if KL divergence is too high or too low
                    if self.adaptive_epsilon and len(epoch_kl_divs) > 0:
                        self.epsilon = self._adjust_epsilon(kl_div)
                    
                    # Calculate surrogate losses with potentially adjusted epsilon
                    surrogate1 = ratio * batch_advantages
                    surrogate2 = tf.clip_by_value(
                        ratio, 1 - self.epsilon, 1 + self.epsilon
                    ) * batch_advantages
                    
                    # PPO-CLIP objective
                    actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                    
                    # Add entropy term for exploration
                    entropy = -tf.reduce_mean(
                        tf.reduce_sum(current_probs_float32 * tf.math.log(current_probs_float32 + 1e-8), axis=1)
                    )
                    actor_loss -= self.entropy_coef * entropy
                
                # Get actor gradients and apply
                actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
                
                # Train critic
                with tf.GradientTape() as tape:
                    # Predict values
                    value_pred = self.critic(batch_states, training=True)
                    value_pred = tf.reshape(value_pred, [-1])
                    
                    # Cast value predictions to float32 to match batch_returns
                    value_pred = tf.cast(value_pred, tf.float32)
                    
                    # Calculate critic loss
                    critic_loss = self.value_coef * tf.reduce_mean(
                        tf.square(batch_returns - value_pred)
                    )
                
                # Get critic gradients and apply
                critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
                
                # Record losses
                history['actor_loss'].append(float(actor_loss))
                history['critic_loss'].append(float(critic_loss))
                history['total_loss'].append(float(actor_loss + critic_loss))
                history['kl_div'].append(float(kl_div))
                
                # Update training step counter for learning rate scheduling
                self.training_steps += 1
                
                # Early stopping if KL divergence is too high
                if self.adaptive_epsilon and float(kl_div) > self.kl_cutoff_factor * 2 * self.kl_target:
                    print(f"Early stopping at epoch {epoch} due to high KL divergence: {float(kl_div):.6f}")
                    break
            
            # Track average KL divergence for this epoch
            avg_kl_div = np.mean(epoch_kl_divs) if epoch_kl_divs else 0
            
            # Get current learning rates
            current_lr = self.get_learning_rates()
            print(f"Epoch {epoch+1}/{epochs}, Avg KL Divergence: {avg_kl_div:.6f}, Epsilon: {self.epsilon:.4f}, " +
                  f"Actor LR: {current_lr['actor_lr']:.6f}, Critic LR: {current_lr['critic_lr']:.6f}")
            
            # Early stopping if KL divergence is too high for the entire epoch
            if self.adaptive_epsilon and avg_kl_div > self.kl_cutoff_factor * 2 * self.kl_target:
                print(f"Early stopping training after epoch {epoch+1} due to high average KL divergence")
                break
        
        # Clear memory after training
        self.clear_memory()
        
        return history
    
    def clear_memory(self):
        """
        Clear trajectory memory
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.action_probs = []
    
    def save_models(self, actor_path, critic_path):
        """Save actor and critic models to disk"""
        self.actor.save(actor_path)
        self.critic.save(critic_path)
    
    def load_models(self, actor_path, critic_path):
        """Load actor and critic models from disk"""
        self.actor = tf.keras.models.load_model(actor_path)
        self.critic = tf.keras.models.load_model(critic_path)
    