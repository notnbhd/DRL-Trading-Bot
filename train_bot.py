import os
import sys

# Add current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the train_agent function
from src.train import train_agent

if __name__ == "__main__":
    # Parse command line arguments if any
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the cryptocurrency trading bot')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading pair symbol (default: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1h',
                        help='Timeframe interval (default: 1h)')
    parser.add_argument('--start-date', type=str, default='2023-05-29',
                        help='Start date for training data (default: 2024-04-01)')
    parser.add_argument('--end-date', type=str, default='2025-05-12',
                        help='End date for training data (default: 2021-07-20)')
    parser.add_argument('--episodes', type=int, default=3000,
                        help='Number of training episodes (default: 4000 as in the paper)')
    parser.add_argument('--trajectory-size', type=int, default=1000,
                        help='Number of steps in each trajectory (default: 1000 as in the paper)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32 as in the paper)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for each update (default: 5 as in the paper)')
    parser.add_argument('--initial-balance', type=float, default=1000,
                        help='Initial balance for trading (default: 100000)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Trading commission rate (default: 0.001 or 0.1%)')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Portion of data to use for testing (default: 0.3)')
    parser.add_argument('--lookback-window', type=int, default=100,
                        help='Lookback window size for observations (default: 100)')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='Frequency to save models during training (default: 1 episodes)')
    parser.add_argument('--fast-train', action='store_true',
                        help='Use a faster training mode with fewer episodes (100) for testing')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU training and use CPU only')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training for faster GPU training')
    parser.add_argument('--resume-from', type=int, default=0,
                        help='Resume training from a specific episode (default: 0, starts from beginning)')
    
    args = parser.parse_args()
    
    # If fast_train is enabled, use fewer episodes
    if args.fast_train:
        episodes = 100
        print("⚠️ FAST TRAINING MODE ENABLED - using only 100 episodes for testing ⚠️")

    else:
        episodes = args.episodes
        print(f"Full training mode with {episodes} episodes")

    
    # GPU configuration
    if args.no_gpu:
        use_gpu = False
        print("❌ GPU training disabled, using CPU only")
    else:
        use_gpu = True
        print("🔍 Checking for available GPUs...")
    
    # Enable mixed precision if requested
    if args.mixed_precision and use_gpu:
        try:
            import tensorflow as tf
            print("Enabling mixed precision training for faster performance")
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision policy set successfully")
        except Exception as e:
            print(f"Warning: Could not enable mixed precision: {e}")
            print("Continuing with default precision settings")
    
    print(f"\nStarting training for {args.symbol} from {args.start_date} to {args.end_date}")
    print(f"Training configuration:")
    print(f"  - Episodes: {episodes}")
    print(f"  - Trajectory size: {args.trajectory_size} steps per episode")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Initial balance: ${args.initial_balance}")
    print(f"  - Commission rate: {args.commission*100}%")
    print(f"  - Lookback window: {args.lookback_window} timeframes")
    print(f"  - Test data split: {args.test_split * 100}%")
    print(f"  - Model save frequency: Every {args.save_freq} episodes")
    print(f"  - GPU training: {'Disabled' if args.no_gpu else 'Enabled'}")
    print(f"  - Mixed precision: {'Enabled' if args.mixed_precision and use_gpu else 'Disabled'}")
    
    # Check if we're resuming from a specific episode
    start_episode = args.resume_from
    if start_episode > 0:
        print(f"🔄 Resuming training from episode {start_episode}")
    
    # Create required directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Run the training with the CNN-LSTM architecture
    train_agent(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        test_split=args.test_split,
        lookback_window_size=args.lookback_window,
        episodes=episodes,
        trajectory_size=args.trajectory_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        initial_balance=args.initial_balance,
        commission=args.commission,
        save_freq=args.save_freq,
        use_gpu=not args.no_gpu,
        start_episode=start_episode
    )