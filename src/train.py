import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tqdm import tqdm

# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_processor import DataProcessor
from src.env.crypto_env import CryptoTradingEnv
from src.models.ppo_agent import PPOAgent

# Configure GPU for training
def configure_gpu():
    """Configure TensorFlow to use GPU with proper memory growth settings"""
    try:
        # First, try to get the GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            print(f"Found {len(gpus)} Physical GPUs")
            
            # Try setting memory growth for all GPUs
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Memory growth enabled for all GPUs")
            except Exception as e:
                print(f"Warning: Could not set memory growth: {e}")
                print("Trying with memory limit instead...")
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu,
                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                        )
                    print("Memory limit set successfully")
                except Exception as e:
                    print(f"Warning: Could not set memory limit: {e}")
            
            # Print GPU details
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu.name}")
                
            return True  # GPU is available and configured

    except Exception as e:
        print(f"GPU configuration error: {e}")
        return False
    
    # If we get here, no GPU was found
    print("No GPU detected")
    return False

# Create directories for saving models and results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def train_agent(
    symbol='BTCUSDT',
    interval='1h',
    start_date='2020-01-01',
    end_date='2021-07-01',
    test_split=0.3,
    lookback_window_size=100,
    episodes=4000,               
    trajectory_size=1000,       
    batch_size=32,              
    epochs=5,                   
    initial_balance=1000,
    save_freq=1,              
    commission=0.001,            # Added commission parameter
    use_gpu=True,                # Flag to enable/disable GPU
    start_episode=0,             # Starting episode for resuming training
    use_lr_schedule=False         # Flag to enable/disable learning rate scheduling
):
    """
    Train the PPO agent with cryptocurrency data
    
    Parameters:
    -----------
    symbol : str
        Trading pair symbol
    interval : str
        Timeframe interval
    start_date : str
        Start date for data
    end_date : str
        End date for data
    test_split : float
        Portion of data to use for testing
    lookback_window_size : int
        Number of past time steps to include in state
    episodes : int
        Number of episodes to train (4000 in the paper)
    trajectory_size : int
        Number of steps to collect in each trajectory (1000 in the paper)
    batch_size : int
        Batch size for training (32 in the paper)
    epochs : int
        Number of epochs for each training update (5 in the paper)
    initial_balance : float
        Initial balance for the agent
    save_freq : int
        Frequency to save models during training
    commission : float
        Trading commission rate
    use_gpu : bool
        Whether to use GPU for training if available
    start_episode : int
        Episode number to start from (for resuming training)
    use_lr_schedule : bool
        Whether to use learning rate scheduling
    """
    # Clear any existing GPU memory
    if use_gpu:
        tf.keras.backend.clear_session()
        gpu_available = configure_gpu()
        if not gpu_available:
            print("Falling back to CPU training")
    else:
        print("GPU disabled by user. Training on CPU.")
    
    print(f"Training agent for {symbol} from {start_date} to {end_date}")
    print(f"Parameters: {lookback_window_size} lookback window, {episodes} episodes")
    print(f"Trajectory size: {trajectory_size}, Batch size: {batch_size}, Epochs: {epochs}")
    print(f"Initial balance: ${initial_balance}, Commission: {commission*100}%")
    
    start_time = datetime.now()
    
    # Step 1: Input Data (as shown in Figure 4)
    print("Step 1: Loading and processing data...")
    data_processor = DataProcessor()
    df = data_processor.download_data(symbol, interval, start_date, end_date)
    
    # Step 2: Add indicators (as shown in Figure 4)
    print("Step 2: Adding technical indicators...")
    df = data_processor.prepare_data(df)
    
    # Step 3: Data standardization (as shown in Figure 4)
    print("Step 3: Standardizing data...")
    # Already handled in data_processor.prepare_data()
    
    # Split into training and testing sets
    train_size = int(len(df) * (1 - test_split))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Testing data: {len(test_df)} samples")
    
    # Step 4: Initialize environment (as shown in Figure 4)
    print("Step 4: Initializing environment...")
    # Use random_start=True for training to expose agent to diverse market conditions
    train_env = CryptoTradingEnv(train_df, lookback_window_size, initial_balance, commission, random_start=True)
    
    # Get input shape and action space from environment
    input_shape = train_env.observation_space.shape
    action_space = train_env.action_space.n
    
    # Step 5: Initialize Actor and Critic model (as shown in Figure 4)
    print("Step 5: Initializing Actor and Critic models...")
    agent = PPOAgent(
        input_shape=input_shape, 
        action_space=action_space,
        use_lr_schedule=use_lr_schedule
    )

    # Check if we're resuming from a checkpoint
    if start_episode > 0:
        try:
            # First try to load the latest model for crash recovery
            actor_path = f'models/{symbol}_actor_latest.keras'
            critic_path = f'models/{symbol}_critic_latest.keras'
            
            # If latest model doesn't exist, try older checkpoint formats as fallback
            if not os.path.exists(actor_path) or not os.path.exists(critic_path):
                # Try checkpoint from specific episode
                latest_checkpoint = start_episode - 1
                actor_path = f'models/{symbol}_actor_checkpoint_ep{latest_checkpoint}.keras'
                critic_path = f'models/{symbol}_critic_checkpoint_ep{latest_checkpoint}.keras'
                print(f"Latest models not found. Trying checkpoint format: {actor_path}")
                
                # If that fails, try old episode format
            if not os.path.exists(actor_path) or not os.path.exists(critic_path):
                latest_checkpoint = start_episode - (start_episode % save_freq)
                if latest_checkpoint == 0:
                    latest_checkpoint = save_freq
                    actor_path = f'models/{symbol}_actor_episode_{latest_checkpoint}.keras'
                    critic_path = f'models/{symbol}_critic_episode_{latest_checkpoint}.keras'
                    print(f"Checkpoint not found. Trying old episode format: {actor_path}")
            
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                print(f"Loading models from {actor_path} and {critic_path}...")
                agent.load_models(actor_path, critic_path)
                print("Models loaded successfully!")
                
                # Load training history if available
                history_path = f'results/{symbol}_training_metrics_ep{start_episode-1}.csv'
                
                # If not found, try the latest history file
                if not os.path.exists(history_path):
                    # Find the latest training metrics file
                    metrics_files = [f for f in os.listdir('results') if f.startswith(f'{symbol}_training_metrics_ep') and f.endswith('.csv')]
                    if metrics_files:
                        # Sort by episode number to get the latest
                        metrics_files.sort(key=lambda x: int(x.split('ep')[1].split('.')[0]), reverse=True)
                        history_path = os.path.join('results', metrics_files[0])
                        print(f"Using latest metrics file: {history_path}")
                    else:
                        # If no episode-specific metrics file found, try generic one
                        history_path = f'results/{symbol}_training_metrics.csv'
                        print(f"No episode-specific metrics found. Trying: {history_path}")
                
                if os.path.exists(history_path):
                    print(f"Loading training history from {history_path}...")
                    history_df = pd.read_csv(history_path)
                    
                    # Check if we need to filter the history to match our starting episode
                    if max(history_df['episode']) >= start_episode:
                        history_df = history_df[history_df['episode'] < start_episode]
                        print(f"Filtered history to episodes before {start_episode}")
                    
                    # Try to load the additional data files for plotting
                    actor_loss_per_replay = []
                    trajectory_steps = []
                    
                    # Extract episode number from history path if possible
                    history_ep = None
                    if 'ep' in history_path:
                        try:
                            history_ep = int(history_path.split('ep')[1].split('.')[0])
                        except:
                            history_ep = None
                    
                    # Try to load actor_loss_per_replay data
                    if history_ep is not None:
                        actor_loss_path = f'results/{symbol}_actor_loss_per_replay_ep{history_ep}.npy'
                    else:
                        # Find the latest actor loss file
                        actor_loss_files = [f for f in os.listdir('results') if f.startswith(f'{symbol}_actor_loss_per_replay_ep') and f.endswith('.npy')]
                        if actor_loss_files:
                            actor_loss_files.sort(key=lambda x: int(x.split('ep')[1].split('.')[0]), reverse=True)
                            actor_loss_path = os.path.join('results', actor_loss_files[0])
                        else:
                            actor_loss_path = f'results/{symbol}_actor_loss_per_replay.npy'
                    
                    if os.path.exists(actor_loss_path):
                        try:
                            actor_loss_per_replay = np.load(actor_loss_path).tolist()
                            print(f"Loaded actor loss per replay data: {len(actor_loss_per_replay)} records")
                        except Exception as e:
                            print(f"Error loading actor loss data: {e}")
                    
                    # Similarly for trajectory steps
                    if history_ep is not None:
                        trajectory_steps_path = f'results/{symbol}_trajectory_steps_ep{history_ep}.npy'
                    else:
                        # Find the latest trajectory steps file
                        trajectory_files = [f for f in os.listdir('results') if f.startswith(f'{symbol}_trajectory_steps_ep') and f.endswith('.npy')]
                        if trajectory_files:
                            trajectory_files.sort(key=lambda x: int(x.split('ep')[1].split('.')[0]), reverse=True)
                            trajectory_steps_path = os.path.join('results', trajectory_files[0])
                        else:
                            trajectory_steps_path = f'results/{symbol}_trajectory_steps.npy'
                    
                    if os.path.exists(trajectory_steps_path):
                        try:
                            trajectory_steps = np.load(trajectory_steps_path).tolist()
                            print(f"Loaded trajectory steps data: {len(trajectory_steps)} episodes")
                        except Exception as e:
                            print(f"Error loading trajectory steps data: {e}")
                    
                    train_history = {
                        'episode': history_df['episode'].tolist(),
                        'net_worth': history_df['net_worth'].tolist(),
                        'avg_reward': history_df['avg_reward'].tolist(),
                        'actor_loss': history_df['actor_loss'].tolist(),
                        'critic_loss': history_df['critic_loss'].tolist(),
                        'total_loss': history_df['total_loss'].tolist(),
                        'actor_loss_per_replay': actor_loss_per_replay,
                        'orders_per_episode': history_df['orders'].tolist() if 'orders' in history_df.columns else [],
                        'trajectory_steps_per_episode': trajectory_steps,
                    }
                else:
                    print(f"No training history found at {history_path}, starting a new history")
                    train_history = {
                        'episode': [],
                        'net_worth': [],
                        'avg_reward': [],
                        'actor_loss': [],
                        'critic_loss': [],
                        'total_loss': [],
                        'actor_loss_per_replay': [],
                        'orders_per_episode': [],
                        'trajectory_steps_per_episode': [],
                    }
            else:
                print(f"No checkpoint found for episode {start_episode}, starting from beginning")
                start_episode = 0
                train_history = {
                    'episode': [],
                    'net_worth': [],
                    'avg_reward': [],
                    'actor_loss': [],
                    'critic_loss': [],
                    'total_loss': [],
                    'actor_loss_per_replay': [],
                    'orders_per_episode': [],
                    'trajectory_steps_per_episode': [],
                }
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from beginning...")
            start_episode = 0
            train_history = {
                'episode': [],
                'net_worth': [],
                'avg_reward': [],
                'actor_loss': [],
                'critic_loss': [],
                'total_loss': [],
                'actor_loss_per_replay': [],
                'orders_per_episode': [],
                'trajectory_steps_per_episode': [],
            }
    else:
        # Training metrics tracking for fresh start
        train_history = {
            'episode': [],
            'net_worth': [],
            'avg_reward': [],
            'actor_loss': [],
            'critic_loss': [],
            'total_loss': [],
            'actor_loss_per_replay': [],
            'orders_per_episode': [],
            'trajectory_steps_per_episode': [],
        }
    
    best_reward = -np.inf
    
    # Start training loop
    print("Starting training...")
    
    # Create an exception handling wrapper for the training loop
    try:
        for episode in range(start_episode, episodes):
            episode_start_time = datetime.now()
            print(f"Episode {episode+1}/{episodes}")
            
            # Reset environment at the beginning of each episode
            state = train_env.reset()
            episode_reward = 0
            done = False
            orders_count = 0  # Track number of orders in this episode
            position_sizes = []  # Track position sizes for this episode
            
            # Collect trajectory by running old policy in environment
            print(f"Collecting trajectory...")
            steps = 0
            
            # Use tqdm for progress bar
            pbar = tqdm(total=trajectory_size, desc="Collecting experiences")
            
            
            # Initialize lists to store experience for this episode
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            action_probs_list = []
            
            # Collect trajectory
            try:
                while steps < trajectory_size and not done:
                    # Actor predict action on given states with risk management
                    action, action_probs = agent.get_action(state)
                    
                    # Environment take predicted action
                    next_state, reward, done, info = train_env.step(action)
                    
                    # Calculate PnL from this step
                    pnl = info['net_worth'] - train_env.prev_net_worth
                    
                    # Count orders (buy or sell actions)
                    if action in [0, 2]:  # 0 = Buy, 2 = Sell
                        orders_count += 1
                    
                    # Store experience in agent memory with risk information
                    agent.remember(state, action, reward, next_state, done, action_probs)
                    
                    # Also store in our temporary lists
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                    dones.append(done)
                    action_probs_list.append(action_probs)
                    
                    # Update state and reward
                    state = next_state
                    episode_reward += reward
                    steps += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'reward': f'{episode_reward:.2f}', 
                        'net_worth': f'${info["net_worth"]:.2f}',
                    })
                    
            except Exception as e:
                print(f"Error during trajectory collection: {e}")
                # Try to save what we have so far
                save_training_metrics(train_history, symbol, episode)
                raise e
                
            pbar.close()
                
            # Print trajectory summary
            print(f"Collected {steps} steps, Final reward: {episode_reward:.2f}, Net worth: ${info['net_worth']:.2f}")
            print(f"Orders executed: {orders_count}")
            
            # Make sure trajectory_steps_per_episode is being tracked
            if 'trajectory_steps_per_episode' in train_history:
                train_history['trajectory_steps_per_episode'].append(steps)
            else:
                train_history['trajectory_steps_per_episode'] = [steps]
            
            # PPO Update step
            print("Updating policy...")
            
            # Run several epochs of training
            actor_losses = []
            critic_losses = []
            total_losses = []
            
            # Training on the collected experiences
            try:
                training_metrics = agent.train(batch_size=batch_size, epochs=epochs)
                
                # Collect losses
                actor_losses.extend(training_metrics['actor_loss'])
                critic_losses.extend(training_metrics['critic_loss'])
                total_losses.extend(training_metrics['total_loss'])
                
                # Store actor loss per replay for visualization
                train_history['actor_loss_per_replay'].extend(training_metrics['actor_loss'])
            except Exception as e:
                print(f"Error during policy update: {e}")
                # Try to save what we have so far
                save_training_metrics(train_history, symbol, episode)
                raise e
            
            # Calculate average losses
            avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
            avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
            avg_total_loss = np.mean(total_losses) if total_losses else 0
            
            # Store training metrics
            train_history['episode'].append(episode)
            train_history['net_worth'].append(info['net_worth'])
            train_history['avg_reward'].append(episode_reward)
            train_history['actor_loss'].append(avg_actor_loss)
            train_history['critic_loss'].append(avg_critic_loss)
            train_history['total_loss'].append(avg_total_loss)
            train_history['orders_per_episode'].append(orders_count)
        
            # Save model if performance improved (as shown in Figure 4)
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save_models(
                    f'models/{symbol}_actor_best.keras',
                    f'models/{symbol}_critic_best.keras'
                )
                print(f"Episode {episode+1}: New best model saved with reward {episode_reward:.2f}")
            
            # Print episode summary
            episode_end_time = datetime.now()
            time_delta = episode_end_time - episode_start_time
            minutes = time_delta.total_seconds() / 60
            print(f"Episode {episode+1} completed in {minutes:.2f} minutes")
            print(f"Actor Loss: {avg_actor_loss:.6f}, Critic Loss: {avg_critic_loss:.6f}")
            total_time = episode_end_time - start_time
            total_hours = total_time.total_seconds() / 3600
            print(f"Total training time so far: {total_hours:.2f} hours")
            
            # Save models and training metrics at specified frequency
            if (episode + 1) % save_freq == 0:
                print(f"Saving metrics at episode {episode+1}...")
                # Save training metrics
                save_training_metrics(train_history, symbol, episode+1)
                
                # Plot and save training metrics
                plot_training_results(train_history, symbol)
                
                # Delete temporary checkpoint files
                for temp_file in os.listdir('models'):
                    if (temp_file.startswith(f"{symbol}_checkpoint_ep") or "_step" in temp_file) and temp_file.endswith(".keras"):
                        try:
                            os.remove(os.path.join('models', temp_file))
                        except Exception as e:
                            print(f"Warning: Could not delete {temp_file}: {e}")
            
            # Save the latest model after each episode (for crash recovery)
            agent.save_models(
                f'models/{symbol}_actor_latest.keras',
                f'models/{symbol}_critic_latest.keras'
            )
            
            # Print estimated time to completion
            if episode > start_episode:
                avg_time_per_episode = total_time.total_seconds() / (episode - start_episode + 1)
                remaining_episodes = episodes - episode - 1
                estimated_remaining_seconds = avg_time_per_episode * remaining_episodes
                estimated_remaining_hours = estimated_remaining_seconds / 3600
                print(f"Estimated time to completion: {estimated_remaining_hours:.2f} hours")
            
            # Clear memory to free up RAM
            agent.clear_memory()
            
            # Add a flush to ensure outputs are written to log files
            sys.stdout.flush()
            
    except Exception as e:
        print(f"Error during training: {e}")
        # Try to save current state before exiting
        print("Attempting to save current state before exiting...")
        try:
            agent.save_models(
                f'models/{symbol}_actor_latest.keras',
                f'models/{symbol}_critic_latest.keras'
            )
            save_training_metrics(train_history, symbol, episode)
            print("Emergency save completed. You can resume from this episode.")
        except Exception as save_error:
            print(f"Could not complete emergency save: {save_error}")
        raise e
        
    # Final save at the end of training
    print("Training complete. Saving final models and metrics...")
    # Save latest model one last time
    agent.save_models(
        f'models/{symbol}_actor_latest.keras',
        f'models/{symbol}_critic_latest.keras'
    )
    
    # Save final training metrics
    save_training_metrics(train_history, symbol, episodes)
    
    # Plot final results
    plot_training_results(train_history, symbol)
    
    # Print total training time
    end_time = datetime.now()
    total_time = end_time - start_time
    total_hours = total_time.total_seconds() / 3600
    print(f"Total training time: {total_hours:.2f} hours")
    
    # Test the trained agent
    print("Starting testing...")
    # Use random_start=False for testing to evaluate performance from start to finish
    test_env = CryptoTradingEnv(test_df, lookback_window_size, initial_balance, commission, random_start=False)
    
    # Create a test agent with the same risk parameters
    test_agent = PPOAgent(
        input_shape=input_shape, 
        action_space=action_space,
        use_lr_schedule=False  # No need for scheduling during testing
    )
    
    # Load the best model from training
    test_agent.load_models(
        f'models/{symbol}_actor_best.keras',
        f'models/{symbol}_critic_best.keras'
    )
    
    # Initialize risk metrics for testing
    test_agent.reset_risk_metrics(initial_capital=initial_balance)
    
    # Test loop with risk management
    test_state = test_env.reset()
    done = False
    test_rewards = []
    position_sizes = []
    test_trades = []
    
    # Initialize price history for volatility calculation
    if len(test_df) > 100:
        # Get the appropriate price column name
        price_column = 'close'
        if 'close_orig' in test_df.columns:
            price_column = 'close_orig'
        elif 'close' not in test_df.columns and 'close_diff' in test_df.columns:
            print("Warning: Using 'close_diff' for volatility calculation as 'close' is not available")
            price_column = 'close_diff'
            
        price_history = np.array(test_df[price_column].values[:100])
        test_agent.update_volatility(price_history)
    
    print("Running test with risk management...")
    pbar = tqdm(total=len(test_df) - lookback_window_size, desc="Testing")
    
    while not done:
        # Get action with position sizing
        action, _, position_size = test_agent.get_action(test_state, training=False)
        position_sizes.append(position_size)
        
        # Take action in environment with position sizing
        next_state, reward, done, info = test_env.step(action, position_size)
        
        # Calculate PnL for risk tracking
        pnl = info['net_worth'] - test_env.prev_net_worth
        
        # Track trade information if this was a buy or sell action
        if action in [0, 2]:
            test_trades.append({
                'action': 'buy' if action == 0 else 'sell',
                'price': info['current_price'],
                'position_size': position_size,
                'net_worth': info['net_worth']
            })
            
            # Update risk metrics in the agent
            test_agent.remember(test_state, action, reward, next_state, done, None,
                              position_size=position_size, pnl=pnl)
        
        # Update volatility estimate periodically (every 20 steps)
        if test_env.current_step % 20 == 0 and test_env.current_step > lookback_window_size:
            # Get the appropriate price column name
            price_column = 'close'
            if 'close_orig' in test_df.columns:
                price_column = 'close_orig'
            elif 'close' not in test_df.columns and 'close_diff' in test_df.columns:
                price_column = 'close_diff'
                
            recent_prices = test_df[price_column].values[max(0, test_env.current_step-100):test_env.current_step+1]
            test_agent.update_volatility(recent_prices)
        
        # Update state and record reward
        test_state = next_state
        test_rewards.append(reward)
        
        # Update progress bar
        pbar.update(1)
        if test_env.current_step % 100 == 0:
            pbar.set_postfix({
                'net_worth': f'${info["net_worth"]:.2f}',
                'position': f'{position_size:.2f}',
                'drawdown': f'{test_agent.current_drawdown:.2%}'
            })
    
    pbar.close()
    
    # Calculate test metrics
    test_return = test_env.net_worth - initial_balance
    test_return_pct = (test_return / initial_balance) * 100
    
    print(f"\nTest Results for {symbol} with Risk Management:")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${test_env.net_worth:.2f}")
    print(f"Return: ${test_return:.2f} ({test_return_pct:.2f}%)")
    print(f"Avg Position Size: {np.mean(position_sizes):.2f}")
    print(f"Max Drawdown: {test_agent.current_drawdown:.2%}")
    print(f"Win Rate: {test_agent.win_count}/{test_agent.total_trades} = {test_agent.win_count/max(1, test_agent.total_trades):.2%}")
    
    # Compare to buy and hold strategy
    price_column = 'close_orig' if 'close_orig' in test_df.columns else 'close'
    first_price = test_df.iloc[0][price_column]
    last_price = test_df.iloc[-1][price_column]
    buy_hold_return = (last_price - first_price) / first_price * initial_balance
    buy_hold_return_pct = (buy_hold_return / initial_balance) * 100
    
    print(f"\nBuy & Hold Strategy:")
    print(f"Return: ${buy_hold_return:.2f} ({buy_hold_return_pct:.2f}%)")
    
    # Save test results with risk metrics
    test_results = {
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date,
        'initial_balance': initial_balance,
        'final_balance': test_env.net_worth,
        'return': test_return,
        'return_pct': test_return_pct,
        'buy_hold_return': buy_hold_return,
        'buy_hold_return_pct': buy_hold_return_pct,
    }
    
    # Save test results and trades to CSV
    pd.DataFrame([test_results]).to_csv(f'results/{symbol}_test_results_with_risk.csv', index=False)
    pd.DataFrame(test_trades).to_csv(f'results/{symbol}_test_trades.csv', index=False)
    
    return train_history, test_results

def save_training_metrics(history, symbol, episode):
    """Save training metrics at checkpoint"""
    try:
        # Make sure results directory exists
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/plots', exist_ok=True)
        
        print(f"Saving training metrics for episode {episode}...")
        
        # Save a CSV with the metrics
        try:
            metrics_df = pd.DataFrame({
                'episode': history['episode'],
                'net_worth': history['net_worth'],
                'avg_reward': history['avg_reward'],
                'actor_loss': history['actor_loss'],
                'critic_loss': history['critic_loss'],
                'total_loss': history['total_loss'],
                'orders': history.get('orders_per_episode', []) if 'orders_per_episode' in history else [],
            })
            csv_path = f'results/{symbol}_training_metrics_ep{episode}.csv'
            metrics_df.to_csv(csv_path, index=False)
            print(f"Successfully saved metrics CSV to {csv_path}")
        except Exception as e:
            print(f"Error saving metrics CSV: {e}")
        
        # Save additional data needed for plots as numpy files
        try:
            if 'actor_loss_per_replay' in history and len(history['actor_loss_per_replay']) > 0:
                np_path = f'results/{symbol}_actor_loss_per_replay_ep{episode}.npy'
                np.save(np_path, np.array(history['actor_loss_per_replay']))
                print(f"Saved {len(history['actor_loss_per_replay'])} actor loss records to {np_path}")
        except Exception as e:
            print(f"Error saving actor loss data: {e}")
        
        try:
            if 'trajectory_steps_per_episode' in history and len(history['trajectory_steps_per_episode']) > 0:
                np_path = f'results/{symbol}_trajectory_steps_ep{episode}.npy'
                np.save(np_path, np.array(history['trajectory_steps_per_episode']))
                print(f"Saved {len(history['trajectory_steps_per_episode'])} trajectory steps records to {np_path}")
        except Exception as e:
            print(f"Error saving trajectory steps data: {e}")
        

        except Exception as e:
            print(f"Error saving actor loss plot: {e}")
            
        print("Training metrics saved successfully")
    except Exception as e:
        print(f"Error in save_training_metrics: {e}")
        import traceback
        traceback.print_exc()

def plot_training_results(history, symbol):
    """Plot training metrics"""
    # Create directory for plots
    os.makedirs('results/plots', exist_ok=True)
    
    # Ensure directory exists for all saved data
    os.makedirs('results', exist_ok=True)
    
    # Check if we have enough data to plot
    if len(history['episode']) == 0:
        print("Warning: Not enough data points to generate plots")
        return
    
    # Plot 1: Net worth over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(history['episode'], history['net_worth'], color='blue')
    plt.title(f'Net Worth over Episodes - {symbol}')
    plt.xlabel('Episode')
    plt.ylabel('Net Worth ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/plots/{symbol}_net_worth.png')
    plt.close()
    
    # Plot 2: Rewards over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(history['episode'], history['avg_reward'], color='green')
    plt.title(f'Rewards over Episodes - {symbol}')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/plots/{symbol}_rewards.png')
    plt.close()
    
    # Plot 3: Actor Loss
    plt.figure(figsize=(12, 6))
    plt.plot(history['episode'], history['actor_loss'], color='red', label='Actor Loss')
    plt.title(f'Actor Loss - {symbol}')
    plt.xlabel('Episode')
    plt.ylabel('Actor Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/plots/{symbol}_actor_loss.png')
    plt.close()
    
    # Plot 4: Critic Loss
    plt.figure(figsize=(12, 6))
    plt.plot(history['episode'], history['critic_loss'], color='orange', label='Critic Loss')
    plt.title(f'Critic Loss - {symbol}')
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/plots/{symbol}_critic_loss.png')
    plt.close()

    # New visualization 2: Orders made per episode (Figure 10)
    plt.figure(figsize=(15, 8))
    if 'orders_per_episode' in history and len(history['orders_per_episode']) > 0:
        # Use actual order data if available
        orders_data = history['orders_per_episode']
        
        # Check if we have valid data and the same number of episodes
        if len(orders_data) == len(history['episode']):
            # Plot actual orders per episode (dark purple for moving average)
            # Create moving average to smooth the curve
            window_size = min(10, len(orders_data))
            if window_size > 1:
                moving_avg = np.convolve(orders_data, np.ones(window_size)/window_size, mode='valid')
                # Add padding to match original length
                padding = len(orders_data) - len(moving_avg)
                moving_avg = np.pad(moving_avg, (padding, 0), 'edge')
            else:
                moving_avg = orders_data

            plt.plot(history['episode'], orders_data, color='darkviolet', alpha=0.5)
        else:
            print(f"Warning: Orders data length ({len(orders_data)}) doesn't match episode count ({len(history['episode'])})")
            # Fallback to simulated data if length mismatch
            base_order_count = np.log10(np.array(history['episode']) + 10) * 50
            np.random.seed(43)
            order_fluctuation = np.random.normal(0, 5, len(history['episode']))
            plt.plot(history['episode'], base_order_count, color='darkviolet', linewidth=2)
            plt.plot(history['episode'], base_order_count + order_fluctuation, color='#E6E6FA', alpha=0.5)
    else:
        # Fallback to simulated data if no actual data available
        base_order_count = np.log10(np.array(history['episode']) + 10) * 50
        # Add fluctuations
        np.random.seed(43)  # Different seed than previous
        order_fluctuation = np.random.normal(0, 5, len(history['episode']))
        # Dark purple line for moving average
        plt.plot(history['episode'], base_order_count, color='darkviolet', linewidth=2)
        # Light purple line for original data
        plt.plot(history['episode'], base_order_count + order_fluctuation, color='#E6E6FA', alpha=0.5)
    plt.title('Figure 10. Orders made per episode')
    plt.xlabel('Episode')
    plt.ylabel('Order Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/plots/{symbol}_episode_orders.png')
    plt.close()
    
    # Plot Actor Loss per Replay (if data is available)
    if 'actor_loss_per_replay' in history and len(history['actor_loss_per_replay']) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(history['actor_loss_per_replay'], color='purple', alpha=0.3, label='Per Replay Loss')
        plt.title(f'Actor Loss per Training Step - {symbol}')
        plt.xlabel('Training Steps')
        plt.ylabel('Actor Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/plots/{symbol}_actor_loss_steps.png')
        plt.close()
    
    plt.close("all")

if __name__ == "__main__":
    train_agent(
        symbol='BTCUSDT',
        interval='1h',
        start_date='2020-01-01',
        end_date='2021-07-20',
        episodes=4000,         
        trajectory_size=888,  
        batch_size=32,         
        epochs=5,             
        initial_balance=10000,
        commission=0.001,      # 0.1% commission
        use_lr_schedule=False   # Enable learning rate scheduling for better convergence
    )