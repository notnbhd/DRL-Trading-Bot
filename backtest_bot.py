import os
import sys

# Add current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Backtester class
from src.backtest import Backtester

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest the cryptocurrency trading bot with CNN-LSTM model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading pair symbol (default: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1h',
                        help='Timeframe interval (default: 1h)')
    parser.add_argument('--start-date', type=str, default='2024-12-12',
                        help='Start date for backtesting (default: 2023-01-01)')
    parser.add_argument('--end-date', type=str, default='2025-05-12',
                        help='End date for backtesting (default: 2024-01-01)')
    parser.add_argument('--lookback-window', type=int, default=100,
                        help='Lookback window size for observations (default: 100)')
    parser.add_argument('--initial-balance', type=float, default=1000,
                        help='Initial balance for backtesting (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Trading commission rate (default: 0.001 or 0.1%)')
    parser.add_argument('--model-path', type=str, default='best_model',
                        help='Path to saved model files (default: models)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for reports (default: results)')
    
    args = parser.parse_args()
    
    # Ensure model files exist
    actor_path = f"{args.model_path}/{args.symbol}_actor_best.pt"
    critic_path = f"{args.model_path}/{args.symbol}_critic_best.pt"
    
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        print(f"ERROR: Model files not found at {actor_path} and {critic_path}")
        print("Please train the model first using train_bot.py or provide the correct model path.")
        sys.exit(1)
    
    print(f"Starting backtest for {args.symbol} from {args.start_date} to {args.end_date}")
    print(f"Using lookback window size {args.lookback_window}, initial balance ${args.initial_balance}")
    print(f"Commission rate: {args.commission*100}%")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Create and run backtester
    backtester = Backtester(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_window_size=args.lookback_window,
        initial_balance=args.initial_balance,
        commission=args.commission,
        model_path=args.model_path
    )
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Generate report
    backtester.generate_report(args.output_dir)
    
    print("\nBacktesting completed successfully!")
    print(f"Results have been saved to {args.output_dir}/")
    print(f"Summary: {results['return_pct']:.2f}% return vs {results['buy_hold_return']:.2f}% buy & hold") 