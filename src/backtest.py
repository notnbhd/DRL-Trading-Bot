import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import accuracy_score
import seaborn as sns
from tqdm import tqdm

from src.env.crypto_env import CryptoTradingEnv
from src.models.ppo_agent import PPOAgent
from src.utils.data_processor import DataProcessor

class Backtester:
    """
    Backtesting class for evaluating trading performance on historical data
    based on the trained PPO agent.
    """
    
    def __init__(
        self,
        symbol='BTCUSDT',
        interval='1h',
        start_date='2021-01-01',
        end_date='2022-01-01',
        lookback_window_size=100,
        initial_balance=100000,
        commission=0.001,
        model_path='models'
    ):
        """
        Initialize the backtester
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        interval : str
            Timeframe interval
        start_date : str
            Start date for backtesting
        end_date : str
            End date for backtesting
        lookback_window_size : int
            Number of past time steps to include in state
        initial_balance : float
            Initial balance for backtesting
        commission : float
            Trading commission rate
        model_path : str
            Path to saved model files
        """
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.model_path = model_path
        
        # Load and prepare data
        self.data_processor = DataProcessor()
        self.df = self._prepare_data()
        
        # Initialize environment with random_start=False for backtesting
        # Backtesting should always evaluate from the beginning to end sequentially
        self.env = CryptoTradingEnv(
            self.df, 
            lookback_window_size=self.lookback_window_size,
            initial_balance=self.initial_balance,
            commission=self.commission,
            random_start=False
        )
        
        # Initialize agent
        self._initialize_agent()
        
        # Results tracking
        self.results = None
    
    def _prepare_data(self):
        """Download and preprocess data for backtesting"""
        print(f"Loading data for {self.symbol} from {self.start_date} to {self.end_date}...")
        
        # Download data
        df = self.data_processor.download_data(
            self.symbol, 
            self.interval, 
            self.start_date, 
            end_str=self.end_date
        )
        
        # Preprocess data
        df = self.data_processor.prepare_data(df)
        
        print(f"Loaded {len(df)} data points.")
        return df
    
    def _initialize_agent(self):
        """Initialize PPO agent with saved models"""
        # Determine input shape from environment
        input_shape = self.env.observation_space.shape
        action_space = self.env.action_space.n
        
        # Create agent
        self.agent = PPOAgent(input_shape, action_space)
        
        # Attempt to load model
        try:
            actor_path = f"{self.model_path}/{self.symbol}_actor_best.keras"
            critic_path = f"{self.model_path}/{self.symbol}_critic_best.keras"
            
            self.agent.load_models(actor_path, critic_path)
            print(f"Loaded model from {actor_path} and {critic_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def run_backtest(self):
        """
        Run backtest on historical data using the trained agent
        
        Returns:
        --------
        Dict of backtest results
        """
        print(f"Running backtest for {self.symbol} from {self.start_date} to {self.end_date}...")
        
        # Reset environment
        state = self.env.reset()
        
        # Tracking metrics
        actions_taken = []
        rewards = []
        done = False
        total_steps = len(self.df) - self.lookback_window_size - 1
        
        # Run through each step and collect metrics
        with tqdm(total=total_steps, desc="Backtesting") as pbar:
            while not done:
                # Get action from agent (using greedy policy)
                action, action_probs = self.agent.get_action(state, training=False)
                
                # Take action in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Record action and reward
                actions_taken.append(action)
                rewards.append(reward)
                
                # Move to next state
                state = next_state
                
                # Update progress bar
                pbar.update(1)
        
        # Calculate additional metrics using trade history
        trade_history = self.env.get_trade_history()
        performance_metrics = self.env.get_performance_metrics()
        
        # Calculate action distribution
        action_dist = {
            'buy': actions_taken.count(0) / len(actions_taken) * 100,
            'hold': actions_taken.count(1) / len(actions_taken) * 100,
            'sell': actions_taken.count(2) / len(actions_taken) * 100
        }
        
        # Calculate profit factor if we have both buys and sells
        profit_factor = 0
        if len(trade_history) > 0 and 'revenue' in trade_history.columns and 'cost' in trade_history.columns:
            total_gains = trade_history[trade_history['type'] == 'sell']['revenue'].sum()
            total_losses = trade_history[trade_history['type'] == 'buy']['cost'].sum()
            if total_losses > 0:
                profit_factor = total_gains / total_losses
        
        # Calculate Sharpe ratio if we have more than 1 day of data
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        
        # Ensure timestamps and history arrays have the same length
        # Note: We need to handle the case where the first element of net_worth_history
        # is the initial balance before any steps are taken
        datetime_index = self.df.index[self.lookback_window_size:]
        
        # If history arrays have one extra element (initial value), remove it for plotting
        net_worth_history = self.env.net_worth_history
        balance_history = self.env.balance_history
        crypto_held_history = self.env.crypto_held_history
        positions_history = self.env.positions_history
        
        # Ensure all arrays are the same length as datetime_index
        if len(net_worth_history) > len(datetime_index):
            net_worth_history = net_worth_history[1:]  # Remove initial value
        
        if len(balance_history) > len(datetime_index):
            balance_history = balance_history[1:]  # Remove initial value
            
        if len(crypto_held_history) > len(datetime_index):
            crypto_held_history = crypto_held_history[1:]  # Remove initial value
            
        if len(positions_history) > len(datetime_index):
            positions_history = positions_history[1:]  # Remove initial value
        
        # Make sure we have the correct price data from the dataframe
        if self.env.has_original_close:
            # Use original prices if we've applied differencing
            price_column = 'close_orig' 
        else:
            price_column = 'close'
            
        price_history = self.df[price_column].values[self.lookback_window_size:]
        
        # Store results
        self.results = {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_balance': self.initial_balance,
            'final_balance': self.env.balance,
            'final_crypto': self.env.crypto_held,
            'final_net_worth': self.env.net_worth,
            'return_pct': (self.env.net_worth / self.initial_balance - 1) * 100,
            'total_trades': len(trade_history) if not trade_history.empty else 0,
            'action_distribution': action_dist,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'max_drawdown': performance_metrics.get('max_drawdown', 0),
            'trade_history': trade_history,
            'net_worth_history': net_worth_history,
            'balance_history': balance_history,
            'crypto_held_history': crypto_held_history,
            'positions_history': positions_history,
            'price_history': price_history,
            'datetime_index': datetime_index,
            'actions_taken': actions_taken,
            'rewards': rewards
        }
        
        print("\nBacktest Results:")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${self.env.balance:.2f}")
        print(f"Final Crypto Held: {self.env.crypto_held:.6f} {self.symbol.replace('USDT', '')}")
        print(f"Final Net Worth: ${self.env.net_worth:.2f}")
        print(f"Return: {self.results['return_pct']:.2f}%")
        print(f"Total Trades: {self.results['total_trades']}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2f}%")
        
        # Compare to buy and hold strategy
        self._compare_to_buy_hold()
        
        return self.results
    
    def _compare_to_buy_hold(self):
        """Compare backtest results to buy and hold strategy"""
        if self.df is None or len(self.df) < 2:
            print("Insufficient data to compare with buy and hold strategy")
            return
        
        # Get price column to use
        price_column = 'close_orig' if self.env.has_original_close else 'close'
        
        # Get first and last price
        first_price = self.df.iloc[self.lookback_window_size][price_column]
        last_price = self.df.iloc[-1][price_column]
        
        # Calculate buy and hold return
        buy_hold_return = (last_price - first_price) / first_price * 100
        buy_hold_profit = self.initial_balance * (1 + buy_hold_return / 100) - self.initial_balance
        
        # Add to results
        self.results['buy_hold_return'] = buy_hold_return
        self.results['buy_hold_profit'] = buy_hold_profit
        self.results['outperformance'] = self.results['return_pct'] - buy_hold_return
        
        print("\nBuy & Hold Comparison:")
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"Strategy Outperformance: {self.results['outperformance']:.2f}%")
    
    def generate_report(self, output_dir='results'):
        """
        Generate detailed backtest report with visualizations
        
        Parameters:
        -----------
        output_dir : str
            Directory to save report files
        """
        if self.results is None:
            print("No backtest results to report. Run backtest first.")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert timestamps to datetime
        timestamps = self.results['datetime_index']
        
        # Generate performance plots
        self._plot_equity_curve(timestamps, output_dir)
        self._plot_trade_positions(timestamps, output_dir)
        self._plot_drawdown(timestamps, output_dir)
        self._plot_action_distribution(output_dir)
        
        # Save trade history to CSV
        if not self.results['trade_history'].empty:
            self.results['trade_history'].to_csv(f"{output_dir}/{self.symbol}_trade_history.csv", index=False)
        
        # Save overall metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': [
                'Symbol', 'Start Date', 'End Date', 'Initial Balance', 'Final Balance',
                'Final Net Worth', 'Return (%)', 'Buy & Hold Return (%)', 'Outperformance (%)',
                'Total Trades', 'Sharpe Ratio', 'Max Drawdown (%)', 'Profit Factor'
            ],
            'Value': [
                self.symbol, self.start_date, self.end_date, self.initial_balance,
                self.results['final_balance'], self.results['final_net_worth'],
                self.results['return_pct'], self.results['buy_hold_return'],
                self.results['outperformance'], self.results['total_trades'],
                self.results['sharpe_ratio'], self.results['max_drawdown'],
                self.results['profit_factor']
            ]
        })
        
        metrics_df.to_csv(f"{output_dir}/{self.symbol}_backtest_metrics.csv", index=False)
        
        print(f"\nBacktest report generated in {output_dir}/")
    
    def _plot_equity_curve(self, timestamps, output_dir):
        """Plot equity curve with buy and hold comparison"""
        plt.figure(figsize=(14, 7))
        
        # Plot net worth - no need to slice with [1:] anymore
        plt.plot(timestamps, self.results['net_worth_history'], label='Strategy Net Worth', linewidth=2)
        
        # Plot buy and hold line
        price_column = 'close_orig' if self.env.has_original_close else 'close'
        initial_price = self.df.iloc[self.lookback_window_size][price_column]
        normalized_prices = self.df[price_column].values[self.lookback_window_size:] / initial_price * self.initial_balance
        plt.plot(timestamps, normalized_prices, label='Buy & Hold', linestyle='--', linewidth=2)
        
        # Add trade markers
        if not self.results['trade_history'].empty:
            buys = self.results['trade_history'][self.results['trade_history']['type'] == 'buy']
            sells = self.results['trade_history'][self.results['trade_history']['type'] == 'sell']
            
            if not buys.empty:
                plt.scatter(buys['time'], buys['net_worth'], color='green', marker='^', s=100, label='Buy')
            if not sells.empty:
                plt.scatter(sells['time'], sells['net_worth'], color='red', marker='v', s=100, label='Sell')
        
        plt.title(f'{self.symbol} Backtest Equity Curve', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Net Worth ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{self.symbol}_equity_curve.png")
        plt.close()
    
    def _plot_trade_positions(self, timestamps, output_dir):
        """Plot cryptocurrency price with trade positions"""
        plt.figure(figsize=(14, 10))
        
        # Create two subplots
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        
        # Plot price on top subplot
        ax1.plot(timestamps, self.results['price_history'], color='blue', linewidth=2)
        ax1.set_title(f'{self.symbol} Price and Positions', fontsize=16)
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add trade markers to price plot
        if not self.results['trade_history'].empty:
            buys = self.results['trade_history'][self.results['trade_history']['type'] == 'buy']
            sells = self.results['trade_history'][self.results['trade_history']['type'] == 'sell']
            
            if not buys.empty:
                ax1.scatter(buys['time'], buys['price'], color='green', marker='^', s=100, label='Buy')
            if not sells.empty:
                ax1.scatter(sells['time'], sells['price'], color='red', marker='v', s=100, label='Sell')
            
            ax1.legend()
        
        # Plot position size on bottom subplot - no need to slice with [1:] anymore
        ax2.plot(timestamps, self.results['crypto_held_history'], color='purple', linewidth=2)
        ax2.set_title(f'Cryptocurrency Holdings', fontsize=16)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel(f'Holdings ({self.symbol.replace("USDT", "")})', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{self.symbol}_positions.png")
        plt.close()
    
    def _plot_drawdown(self, timestamps, output_dir):
        """Plot drawdown analysis"""
        plt.figure(figsize=(14, 7))
        
        # Calculate drawdown series
        net_worths = np.array(self.results['net_worth_history'])
        peak = np.maximum.accumulate(net_worths)
        drawdown = (peak - net_worths) / peak * 100
        
        # Plot drawdown
        plt.plot(timestamps, drawdown, color='red', linewidth=2)
        plt.fill_between(timestamps, drawdown, 0, color='red', alpha=0.3)
        
        plt.title(f'{self.symbol} Drawdown Analysis', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Invert y-axis for better visualization
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{self.symbol}_drawdown.png")
        plt.close()
    
    def _plot_action_distribution(self, output_dir):
        """Plot distribution of actions taken"""
        plt.figure(figsize=(10, 6))
        
        # Get action distribution
        actions = ['Buy', 'Hold', 'Sell']
        percentages = [
            self.results['action_distribution']['buy'],
            self.results['action_distribution']['hold'],
            self.results['action_distribution']['sell']
        ]
        
        # Create bar chart
        plt.bar(actions, percentages, color=['green', 'blue', 'red'])
        
        # Add percentage labels on top of bars
        for i, p in enumerate(percentages):
            plt.text(i, p + 1, f'{p:.1f}%', ha='center')
        
        plt.title(f'{self.symbol} Action Distribution', fontsize=16)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{self.symbol}_action_distribution.png")
        plt.close()

# Run backtest if executed directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest the cryptocurrency trading bot')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading pair symbol (default: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1h',
                        help='Timeframe interval (default: 1h)')
    parser.add_argument('--start-date', type=str, default='2021-01-01',
                        help='Start date for backtest (default: 2021-01-01)')
    parser.add_argument('--end-date', type=str, default='2022-01-01',
                        help='End date for backtest (default: 2022-01-01)')
    parser.add_argument('--initial-balance', type=float, default=10000,
                        help='Initial balance (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Trading commission rate (default: 0.001)')
    parser.add_argument('--model-path', type=str, default='models',
                        help='Path to model files (default: models)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for reports (default: results)')
    
    args = parser.parse_args()
    
    # Create and run backtester
    backtester = Backtester(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance,
        commission=args.commission,
        model_path=args.model_path
    )
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Generate report
    backtester.generate_report(args.output_dir) 