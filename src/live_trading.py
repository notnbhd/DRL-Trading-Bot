import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_processor import DataProcessor
from src.models.ppo_agent import PPOAgent

class LiveTrader:
    """Live trading implementation using the trained PPO agent"""
    
    def __init__(
        self,
        api_key="",
        api_secret="",
        symbol="BTCUSDT",
        interval="1h",
        initial_balance=10000,
        commission=0.001,
        lookback_window_size=100,
        model_path="models",
        test_mode=True
    ):
        """
        Initialize the live trader
        
        Parameters:
        -----------
        api_key : str
            Binance API key
        api_secret : str
            Binance API secret
        symbol : str
            Trading pair symbol
        interval : str
            Timeframe interval
        lookback_window_size : int
            Number of past time steps to include in state
        model_path : str
            Path to saved models
        test_mode : bool
            Whether to run in test mode (no real trades)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.interval = interval
        self.lookback_window_size = lookback_window_size
        self.model_path = model_path
        self.test_mode = test_mode
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Connect to Binance
        self.client = Client(api_key, api_secret)
        
        # Initialize data processor
        self.data_processor = DataProcessor()
        
        # Load model
        self._load_model()
        
        # Track positions and balance
        self.position = 0  # 0: No position, 1: Long
        self.balance = 0
        self.crypto_held = 0
        self.last_action = None
        self.trade_history = []
    
    def _load_model(self):
        """Load the trained model and scaler"""
        # Get sample data to determine input shape
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.lookback_window_size)
        
        sample_df = self.data_processor.download_data(
            self.symbol,
            self.interval,
            start_time.strftime('%Y-%m-%d'),
            end_time.strftime('%Y-%m-%d')
        )
        
        # Load the scaler fitted during training
        scaler_path = f"{self.model_path}/{self.symbol}_scaler.joblib"
        self.data_processor.load_scaler(scaler_path)
        
        # Process sample data to determine expected input shape (use training scaler)
        processed_df = self.data_processor.prepare_data(sample_df, fit_scaler=False)
        
        # Determine input shape from sample data
        input_shape = (self.lookback_window_size, processed_df.shape[1])  # +2 for balance and crypto held
        action_space = 3  # Buy, Hold, Sell
        
        # Initialize agent
        self.agent = PPOAgent(input_shape, action_space)
        
        # Load trained models
        try:
            self.agent.load_models(
                f"{self.model_path}/{self.symbol}_actor_best.keras",
                f"{self.model_path}/{self.symbol}_critic_best.keras"
            )
            print(f"Model loaded for {self.symbol}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _get_account_info(self):
        """Get account balance information"""
        if self.test_mode:
            print("Test mode: Using simulated account balance")
            return {'balance': 10000, 'crypto': 0}
        
        try:
            account = self.client.get_account()
            
            # Get USDT balance
            usdt_balance = 0
            for asset in account['balances']:
                if asset['asset'] == 'USDT':
                    usdt_balance = float(asset['free'])
            
            # Get crypto balance
            crypto_symbol = self.symbol.replace('USDT', '')
            crypto_balance = 0
            for asset in account['balances']:
                if asset['asset'] == crypto_symbol:
                    crypto_balance = float(asset['free'])
            
            return {'balance': usdt_balance, 'crypto': crypto_balance}
        except BinanceAPIException as e:
            print(f"Error getting account info: {e}")
            return {'balance': 0, 'crypto': 0}
    
    def _get_current_price(self):
        """Get current price of the symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            print(f"Error getting current price: {e}")
            return None
    
    def _execute_trade(self, action):
        """Execute a trade based on the predicted action"""
        current_price = self._get_current_price()
        
        if current_price is None:
            print("Could not get current price. Skipping trade.")
            return False
        
        account_info = self._get_account_info()
        self.balance = account_info['balance']
        self.crypto_held = account_info['crypto']
        
        if action == 0 and self.position == 0:  # Buy
            if self.test_mode:
                print(f"TEST MODE - BUY signal: Would buy {self.symbol} at {current_price}")
                
                # Apply formula [4]: Amount bought = Current net worth / Current crypto closing price
                net_worth = self.balance
                crypto_to_buy = net_worth / current_price
                
                # Update position tracking
                self.crypto_held = crypto_to_buy
                self.balance = 0
                self.position = 1
                
                self.last_action = {
                    'type': 'buy', 
                    'price': current_price, 
                    'time': datetime.now(),
                    'amount': crypto_to_buy,
                    'cost': net_worth
                }
                self.trade_history.append(self.last_action)
                return True
            else:
                try:
                    # Calculate how much to buy using formula [4]
                    net_worth = self.balance
                    amount = net_worth / current_price
                    
                    # Execute market buy
                    order = self.client.order_market_buy(
                        symbol=self.symbol,
                        quantity=self._format_quantity(amount)
                    )
                    
                    print(f"BUY order executed: {order}")
                    self.position = 1
                    self.last_action = {
                        'type': 'buy',
                        'price': current_price,
                        'time': datetime.now(),
                        'amount': amount,
                        'cost': net_worth,
                        'order': order
                    }
                    self.trade_history.append(self.last_action)
                    return True
                except BinanceAPIException as e:
                    print(f"Error executing buy order: {e}")
                    return False
        
        elif action == 2 and self.position == 1:  # Sell
            if self.test_mode:
                print(f"TEST MODE - SELL signal: Would sell {self.symbol} at {current_price}")
                
                # Apply formula [5]: Amount sold = Current crypto amount held × Current crypto closing price
                sell_amount = self.crypto_held * current_price
                
                # Update position tracking
                self.balance += sell_amount
                self.crypto_held = 0
                self.position = 0
                
                self.last_action = {
                    'type': 'sell',
                    'price': current_price,
                    'time': datetime.now(),
                    'amount': self.crypto_held,
                    'revenue': sell_amount
                }
                self.trade_history.append(self.last_action)
                return True
            else:
                try:
                    # Sell using formula [5]
                    # Execute market sell of all holdings
                    order = self.client.order_market_sell(
                        symbol=self.symbol,
                        quantity=self._format_quantity(self.crypto_held)
                    )
                    
                    sell_amount = self.crypto_held * current_price
                    print(f"SELL order executed: {order}")
                    self.position = 0
                    self.last_action = {
                        'type': 'sell',
                        'price': current_price,
                        'time': datetime.now(),
                        'amount': self.crypto_held,
                        'revenue': sell_amount,
                        'order': order
                    }
                    self.trade_history.append(self.last_action)
                    return True
                except BinanceAPIException as e:
                    print(f"Error executing sell order: {e}")
                    return False
        
        return False
    
    def _format_quantity(self, quantity):
        """Format quantity according to exchange requirements"""
        # Get symbol info to determine the precision
        try:
            symbol_info = self.client.get_symbol_info(self.symbol)
            
            # Extract the step size
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    precision = int(round(-np.log10(step_size)))
                    
                    # Format quantity with appropriate precision
                    return "{:.{}f}".format(quantity - (quantity % step_size), precision)
            
            # If no LOT_SIZE filter found, use default precision
            return "{:.6f}".format(quantity)
            
        except BinanceAPIException as e:
            print(f"Error getting symbol info: {e}")
            # Use default precision as fallback
            return "{:.6f}".format(quantity)
    
    def _prepare_state(self, df):
        """Prepare state for model prediction"""
        # Ensure we have enough data
        if len(df) < self.lookback_window_size:
            raise ValueError(f"Not enough data for lookback window. Need {self.lookback_window_size} rows.")
        
        # Get the latest lookback_window_size data points
        recent_data = df.iloc[-self.lookback_window_size:].copy()
        
        # Create state with additional features (balance and crypto held)
        state = np.zeros((self.lookback_window_size, df.shape[1]))
        
        # Fill with market data
        state = recent_data.values
        
        # Add normalized crypto held and balance
        # net_worth = self.balance + (self.crypto_held * self._get_current_price())
        # for i in range(self.lookback_window_size):
        #     state[i, -2] = self.crypto_held / (1 + self.crypto_held)  # Normalize crypto held
        #     state[i, -1] = self.balance / net_worth if net_worth > 0 else 0  # Normalize balance
        
        return state
    
    def run(self, interval_seconds=3600, max_iterations=None):
        """
        Run the live trading bot
        
        Parameters:
        -----------
        interval_seconds : int
            Time between trading decisions in seconds
        max_iterations : int, optional
            Maximum number of iterations to run (None for infinite)
        """
        iteration = 0
        
        print(f"Starting live trading for {self.symbol}")
        print(f"Mode: {'TEST' if self.test_mode else 'LIVE'}")
        print(f"Interval: {self.interval} ({interval_seconds} seconds)")
        
        # Get initial account info
        account_info = self._get_account_info()
        self.balance = account_info['balance']
        self.crypto_held = account_info['crypto']
        
        if self.crypto_held > 0:
            self.position = 1
        else:
            self.position = 0
        
        print(f"Initial balance: {self.balance} USDT")
        print(f"Initial {self.symbol.replace('USDT', '')}: {self.crypto_held}")
        
        # Trading loop
        while True:
            try:
                # Check if max iterations reached
                if max_iterations is not None and iteration >= max_iterations:
                    print(f"Reached maximum iterations ({max_iterations}). Stopping.")
                    break
                
                # Get historical data
                end_time = datetime.now()
                start_time = end_time - timedelta(days=self.lookback_window_size + 5)  # Add buffer
                
                df = self.data_processor.download_data(
                    self.symbol,
                    self.interval,
                    start_time.strftime('%Y-%m-%d'),
                    end_time.strftime('%Y-%m-%d')
                )
                
                # Process data (use training scaler, don't re-fit)
                processed_df = self.data_processor.prepare_data(df, fit_scaler=False)
                
                # Prepare state
                state = self._prepare_state(processed_df[['close_diff', 'rsi', 'atr', 'cmf']])
                
                # Get action prediction
                action, _ = self.agent.get_action(state, training=False)
                
                # Convert action to string for logging
                action_str = "BUY" if action == 0 else "HOLD" if action == 1 else "SELL"
                
                # Get current price
                current_price = self._get_current_price()
                
                # Print prediction
                print(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
                print(f"Current price: {current_price} USDT")
                print(f"Position: {'LONG' if self.position == 1 else 'NONE'}")
                print(f"Prediction: {action_str}")
                
                # Execute trade based on prediction
                if (action == 0 and self.position == 0) or (action == 2 and self.position == 1):
                    trade_executed = self._execute_trade(action)
                    if trade_executed:
                        print(f"Trade executed: {action_str}")
                        
                        # Update account info after trade
                        account_info = self._get_account_info()
                        self.balance = account_info['balance']
                        self.crypto_held = account_info['crypto']
                        
                        print(f"Updated balance: {self.balance} USDT")
                        print(f"Updated {self.symbol.replace('USDT', '')}: {self.crypto_held}")
                
                # Increment iteration counter
                iteration += 1
                
                # Wait for next interval
                print(f"Waiting {interval_seconds} seconds until next trading decision...")
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("\nTrading stopped by user.")
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait a minute before trying again
        
        # Print trading summary
        print("\n--- Trading Summary ---")
        print(f"Total trades: {len(self.trade_history)}")
        
        # Calculate profit/loss if any trades were made
        if len(self.trade_history) > 0:
            buy_trades = [t for t in self.trade_history if t['type'] == 'buy']
            sell_trades = [t for t in self.trade_history if t['type'] == 'sell']
            
            total_bought = sum([t['price'] for t in buy_trades])
            total_sold = sum([t['price'] for t in sell_trades])
            
            if total_bought > 0:
                profit_loss = ((total_sold / total_bought) - 1) * 100
                print(f"Profit/Loss: {profit_loss:.2f}%")
        
        # Save trading history
        if len(self.trade_history) > 0:
            history_df = pd.DataFrame(self.trade_history)
            history_df.to_csv(f"results/{self.symbol}_trade_history.csv", index=False)
            print(f"Trade history saved to results/{self.symbol}_trade_history.csv")

if __name__ == "__main__":
    # Configuration
    API_KEY = ""  # Insert your Binance API key
    API_SECRET = ""  # Insert your Binance API secret
    TEST_MODE = True  # Set to False for real trading
    
    # Create and run trader
    trader = LiveTrader(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol="BTCUSDT",
        interval="1h",
        lookback_window_size=100,
        test_mode=TEST_MODE
    )
    
    # Run with a trading decision every hour, maximum 24 iterations (1 day)
    trader.run(interval_seconds=3600, max_iterations=24) 