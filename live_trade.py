import os
import sys

# Add current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the LiveTrader class
from src.live_trading import LiveTrader

if __name__ == "__main__":
    # Parse command line arguments if any
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the cryptocurrency trading bot in live or test mode using CNN-LSTM model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading pair symbol (default: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1h',
                        help='Timeframe interval (default: 1h)')
    parser.add_argument('--api-key', type=str, default='',
                        help='Binance API key')
    parser.add_argument('--api-secret', type=str, default='',
                        help='Binance API secret')
    parser.add_argument('--test-mode', action='store_true', default=True,
                        help='Run in test mode without real trades (default: True)')
    parser.add_argument('--max-iterations', type=int, default=24,
                        help='Maximum number of trading iterations (default: 24)')
    parser.add_argument('--interval-seconds', type=int, default=3600,
                        help='Seconds between trading decisions (default: 3600)')
    parser.add_argument('--initial-balance', type=float, default=10000,
                        help='Initial balance for test mode (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Trading commission rate (default: 0.001 or 0.1%)')
    parser.add_argument('--lookback-window', type=int, default=100,
                        help='Lookback window size for observations (default: 100)')
    parser.add_argument('--model-path', type=str, default='models',
                        help='Path to saved model files (default: models)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save trading results (default: results)')
    
    args = parser.parse_args()
    
    # Create required directories
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting live trading for {args.symbol}")
    print(f"Mode: {'TEST' if args.test_mode else 'LIVE'}")
    print(f"Interval: {args.interval} ({args.interval_seconds} seconds)")
    
    if args.test_mode:
        print(f"Initial balance: ${args.initial_balance}")
    
    # Make sure model files exist before attempting to load them
    actor_path = f"{args.model_path}/{args.symbol}_actor_best.pt"
    critic_path = f"{args.model_path}/{args.symbol}_critic_best.pt"
    
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        print(f"ERROR: Model files not found at {actor_path} and {critic_path}")
        print("Please train the model first using train_bot.py or provide the correct model path.")
        sys.exit(1)
    
    # Create and run trader with CNN-LSTM model
    trader = LiveTrader(
        api_key=args.api_key,
        api_secret=args.api_secret,
        symbol=args.symbol,
        interval=args.interval,
        lookback_window_size=args.lookback_window,
        model_path=args.model_path,
        test_mode=args.test_mode,
        initial_balance=args.initial_balance,
        commission=args.commission,
    )
    
    # Run with specified parameters
    trader.run(interval_seconds=args.interval_seconds, max_iterations=args.max_iterations) 