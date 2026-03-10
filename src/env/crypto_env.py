import gym
import numpy as np
import pandas as pd
from gym import spaces
from collections import deque

class CryptoTradingEnv(gym.Env):
    """A cryptocurrency trading environment for OpenAI gym based on research paper"""
    
    def __init__(self, df, lookback_window_size=100, initial_balance=1000, commission=0.001, random_start=True):
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission  # Trading commission fee (0.1%)
        self.random_start = random_start  # Whether to randomize starting position
        self.current_step = self.lookback_window_size
        
        # Check if we have the original and differenced price data
        self.has_original_close = 'close_orig' in self.df.columns
        if not self.has_original_close:
            raise ValueError("DataFrame must contain 'close_orig' column for trading calculations")
        
        # Define our 4 specific features for the model input
        self.feature_columns = ['close_diff', 'rsi', 'atr', 'cmf']
        
        # Verify all required features exist for the model
        missing_features = [col for col in self.feature_columns if col not in self.df.columns]
        if missing_features:
            raise ValueError(f"Missing required features in DataFrame: {missing_features}")
        
        # Actions: Buy (0), Hold (1), Sell (2)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 4 specific indicators (100, 4)
        # These are the only features passed to the CNN-LSTM model
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(lookback_window_size, len(self.feature_columns)), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self):
        """
        Reset the environment to a starting position.
        If random_start is True, choose a random starting point in the dataset.
        This ensures diverse training scenarios across different market conditions.
        """
        # Randomize starting position during training to expose agent to diverse scenarios
        if self.random_start:
            # Leave enough room for a meaningful episode (at least 100 steps)
            max_start = len(self.df) - 100
            self.current_step = np.random.randint(self.lookback_window_size, max_start)
        else:
            # For evaluation/backtesting, always start from the beginning
            self.current_step = self.lookback_window_size
        
        # Reset balance, holdings, net worth
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        
        # Reset history and metrics
        self.trades = []
        self.net_worth_history = [self.initial_balance]
        self.balance_history = [self.initial_balance]
        self.crypto_held_history = [0]
        self.returns_history = [0]
        
        # Track positions for visualization
        self.position = 0  # 0: no position, 1: long position
        self.positions_history = [0]
        
        # Get first observation
        return self._next_observation()
    
    def _next_observation(self):
        """
        Get observation of current step with lookback window using only the 4 specified indicators.
        Note: original close price (close_orig) is kept separately for trading calculations.
        """
        # Create a frame for the 4 specific features that will be fed to the CNN-LSTM model
        frame = np.zeros((self.lookback_window_size, len(self.feature_columns)))
        
        # Update the observation with data from lookback window
        for i in range(self.lookback_window_size):
            current_idx = self.current_step - self.lookback_window_size + i
            
            # Extract only the features needed for the model input
            for j, column in enumerate(self.feature_columns):
                frame[i, j] = self.df.iloc[current_idx][column]
            
        return frame
    
    def step(self, action):
        # Get current price from the original closing price
        current_price = self._get_current_price()
        
        # Take action
        self._take_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Update previous net worth (before calculating new net worth)
        self.prev_net_worth = self.net_worth
        
        # Update net worth with current price
        self.net_worth = self.balance + self.crypto_held * self._get_current_price()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update history
        self.net_worth_history.append(self.net_worth)
        self.balance_history.append(self.balance)
        self.crypto_held_history.append(self.crypto_held)
        self.positions_history.append(self.position)
        
        # Calculate return
        current_return = (self.net_worth / self.initial_balance - 1) * 100
        self.returns_history.append(current_return)
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        # Get next observation
        obs = self._next_observation()
        
        # Create info dictionary with metrics
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'current_price': current_price,
            'return_pct': current_return,
            'position': self.position
        }
        
        return obs, reward, done, info
    
    def _take_action(self, action, price):
        """Execute the specified action"""
        
        if action == 0:  # Buy
            self._buy_crypto(price)
        elif action == 2:  # Sell
            self._sell_crypto(price)
        # Else: Hold - do nothing
    
    def _get_current_price(self):
        """Get the current closing price - use original price if available"""
        if self.has_original_close:
            return self.df.iloc[self.current_step]['close_orig']
        else:
            return self.df.iloc[self.current_step]['close']
    
    def _buy_crypto(self, price):
        """
        Execute buy action:
        Amount bought = Current net worth / Current crypto closing price
        """
        if self.balance > 0:
            # Calculate amount to buy including commission
            buy_amount = self.balance / (1 + self.commission)
            crypto_bought = buy_amount / price
            
            # Apply commission
            transaction_cost = buy_amount + (buy_amount * self.commission)
            if transaction_cost > self.balance:
                transaction_cost = self.balance
                buy_amount = self.balance / (1 + self.commission)
                crypto_bought = buy_amount / price
            
            # Update holdings
            self.crypto_held = crypto_bought
            self.balance = self.balance - transaction_cost
            self.position = 1  # Long position
            
            # Record trade with timestamp
            self.trades.append({
                'step': self.current_step,
                'time': self.df.index[self.current_step],
                'type': 'buy',
                'price': price,
                'amount': crypto_bought,
                'cost': transaction_cost,
                'balance_after': self.balance,
                'crypto_after': self.crypto_held,
                'net_worth': self.balance + (self.crypto_held * price)
            })
            
    def _sell_crypto(self, price):
        """
        Execute sell action:
        Amount sold = Current crypto amount held × Current crypto closing price
        """
        if self.crypto_held > 0:
            # Calculate the amount from selling
            sell_amount = self.crypto_held * price
            
            # Apply commission
            transaction_fee = sell_amount * self.commission
            sell_amount -= transaction_fee
            
            # Update balance and holdings
            self.balance += sell_amount
            
            # Record trade with detailed metrics
            self.trades.append({
                'step': self.current_step,
                'time': self.df.index[self.current_step],
                'type': 'sell',
                'price': price,
                'amount': self.crypto_held,
                'revenue': sell_amount,
                'fee': transaction_fee,
                'balance_after': self.balance,
                'crypto_after': 0,
                'net_worth': self.balance
            })
            
            self.crypto_held = 0
            self.position = 0  # No position
    
    def _calculate_reward(self):
        """
        Calculate reward based on the change in portfolio value
        r_t = (v_t - v_{t-1}) / v_{t-1}
        
        Where:
        v_t is the portfolio value at time t
        v_{t-1} is the portfolio value at time t-1
        """
        if self.prev_net_worth > 0:
            reward = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        else:
            reward = 0
        
        return reward
    
    def get_trade_history(self):
        """Return the trade history as a DataFrame for analysis"""
        if len(self.trades) == 0:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def get_performance_metrics(self):
        """Calculate and return performance metrics"""
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'return_pct': 0,
                'max_drawdown': 0
            }
        
        # Calculate metrics
        trade_df = self.get_trade_history()
        buy_trades = trade_df[trade_df['type'] == 'buy']
        sell_trades = trade_df[trade_df['type'] == 'sell']
        
        # Calculate returns and drawdowns
        returns = np.array(self.returns_history)
        net_worths = np.array(self.net_worth_history)
        cummax = np.maximum.accumulate(net_worths)
        drawdowns = (cummax - net_worths) / cummax
        
        return {
            'total_trades': len(buy_trades) + len(sell_trades),
            'final_balance': self.balance,
            'final_net_worth': self.net_worth,
            'return_pct': (self.net_worth / self.initial_balance - 1) * 100,
            'max_drawdown': np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(8760) if len(returns) > 1 else 0
        }
            
    def render(self, mode='human'):
        """Render the current state of the environment"""
        profit = self.net_worth - self.initial_balance
        return_pct = (self.net_worth / self.initial_balance - 1) * 100
        
        print(f'Step: {self.current_step} / {len(self.df) - 1}')
        print(f'Price: {self._get_current_price():.2f}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Crypto held: {self.crypto_held:.6f}')
        print(f'Net Worth: {self.net_worth:.2f}')
        print(f'Profit: {profit:.2f} ({return_pct:.2f}%)')
        print(f'Position: {"LONG" if self.position == 1 else "NONE"}')
        
        return 