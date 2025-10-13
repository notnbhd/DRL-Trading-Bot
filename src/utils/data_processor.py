import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class DataProcessor:
    """Process cryptocurrency data for the trading bot"""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1)) 
    
    def download_data(self, symbol, interval, start_str, end_str=None, source='binance'):
        """
        Download historical data from the specified source
        
        Parameters:
        -----------
        symbol : str
            The trading pair symbol (e.g., 'BTCUSDT')
        interval : str
            The timeframe interval (e.g., '1h', '1d')
        start_str : str
            Start date in format 'YYYY-MM-DD'
        end_str : str, optional
            End date in format 'YYYY-MM-DD'
        source : str, optional
            Data source ('binance' by default)
            
        Returns:
        --------
        DataFrame containing the historical data
        """
        if source == 'binance':
            from binance.client import Client
            api_key = os.getenv('API_KEY')
            api_secret = os.getenv('API_SECRET')
            client = Client(api_key, api_secret)
            
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str
            )
            
            # Create DataFrame
            df = pd.DataFrame(
                klines,
                columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ]
            )
            
            # Convert to numeric values
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert price and volume columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Keep only essential columns
            df = df[numeric_columns]
            
            return df
        else:
            raise ValueError(f"Data source '{source}' not supported")
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the DataFrame as specified in Table 1:
        1. Relative Strength Index (RSI)
        2. Normalized Average True Range (ATR)
        3. Chaikin Money Flow (CMF)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        DataFrame with added technical indicators
        """
        # Copy to avoid modifying the original
        df_processed = df.copy()
        
        # 1. RSI (Relative Strength Index) - "Relative strength index indicator" in Table 1
        rsi_indicator = RSIIndicator(close=df_processed['close'], window=14)
        df_processed['rsi'] = rsi_indicator.rsi()
        
        # 2. ATR (Average True Range) - "Normalised average true range indicator" in Table 1
        atr_indicator = AverageTrueRange(high=df_processed['high'], low=df_processed['low'], 
                                        close=df_processed['close'], window=14)
        # Get ATR values
        atr_values = atr_indicator.average_true_range()
        
        # Normalize ATR by the closing price to make it "Normalised average true range"
        df_processed['atr'] = atr_values
        
        # 3. CMF (Chaikin Money Flow) - Additional indicator for money flow analysis
        # Calculate Money Flow Multiplier: ((Close - Low) - (High - Close)) / (High - Low)
        df_processed['mf_multiplier'] = ((df_processed['close'] - df_processed['low']) - 
                                         (df_processed['high'] - df_processed['close'])) / \
                                        (df_processed['high'] - df_processed['low'] + 1e-10)  # Avoid division by zero
        
        # Calculate Money Flow Volume: Money Flow Multiplier * Volume
        df_processed['mf_volume'] = df_processed['mf_multiplier'] * df_processed['volume']
        
        # Calculate 20-period Chaikin Money Flow: Sum(Money Flow Volume) / Sum(Volume)
        df_processed['cmf'] = df_processed['mf_volume'].rolling(window=20).sum() / \
                              df_processed['volume'].rolling(window=20).sum()
        
        # Clean up intermediate columns
        df_processed.drop(['mf_multiplier', 'mf_volume'], axis=1, inplace=True)
        
        # Drop rows with NaN values (usually at the beginning due to indicators calculation)
        df_processed.dropna(inplace=True)
        
        return df_processed
    
    def apply_difference(self, df):
        """
        Apply differencing to make price data stationary as recommended in the paper
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with price data
            
        Returns:
        --------
        DataFrame with differenced price data
        """
        df_diff = df.copy()
        
        # Store original close price for reference (will be needed in the environment)
        df_diff['close_orig'] = df_diff['close']
        
        # Apply first-order differencing to close price to remove trend
        df_diff['close_diff'] = df_diff['close'].diff()
        
        # Keep only the columns we need
        columns_to_keep = ['close_orig', 'close_diff', 'rsi', 'atr', 'cmf']
        columns_to_keep = [col for col in columns_to_keep if col in df_diff.columns]
        
        # Drop all other columns
        for col in df_diff.columns:
            if col not in columns_to_keep:
                df_diff.drop(col, axis=1, inplace=True)
        
        # Drop the first row with NaN values from differencing
        df_diff.dropna(inplace=True)
        
        return df_diff
    
    def normalize_data(self, df):
        """
        Normalize data using Min-Max scaling to the range [-1, 1] as mentioned in the paper
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with features
            
        Returns:
        --------
        DataFrame with normalized features
        """
        df_normalized = df.copy()
        
        # Store original close price and non-differenced columns separately
        # These will be excluded from normalization to preserve their original values
        preserve_columns = ['close_orig']
        normalized_columns = [col for col in df_normalized.columns if col not in preserve_columns]
        
        # Store column names
        columns = normalized_columns
        
        # Fit and transform only the columns to be normalized
        normalized_data = self.scaler.fit_transform(df_normalized[normalized_columns])
        
        # Convert back to DataFrame with proper indexing
        normalized_df = pd.DataFrame(normalized_data, columns=columns, index=df_normalized.index)
        
        # Add back preserved columns
        for col in preserve_columns:
            if col in df_normalized.columns:
                normalized_df[col] = df_normalized[col]
        
        return normalized_df
    
    def prepare_data(self, df, add_indicators=True, apply_diff=True, normalize=True):
        """
        Prepare data for training by applying all preprocessing steps as described in the paper.
        Creates input states with 100 hours of market information containing:
        - Closing price
        - Relative strength index indicator
        - Normalized average true range indicator
        - Chaikin money flow indicator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw DataFrame with OHLCV data
        add_indicators : bool, optional
            Whether to add technical indicators
        apply_diff : bool, optional
            Whether to apply differencing to make data stationary
        normalize : bool, optional
            Whether to normalize data for faster training
            
        Returns:
        --------
        Preprocessed DataFrame ready for training
        """
        processed_df = df.copy()
        
        # Add technical indicators
        if add_indicators:
            processed_df = self.add_technical_indicators(processed_df)
        
        # Apply differencing to make data stationary
        if apply_diff:
            processed_df = self.apply_difference(processed_df)
        
        # Normalize data to range [-1, 1]
        if normalize:
            processed_df = self.normalize_data(processed_df)
        
        print(f"Prepared data with features: {processed_df.columns.tolist()}")
        print(f"Data includes lookback window of {100} hours with closing price and technical indicators")
        
        return processed_df
    
    def plot_data_comparison(self, original_df, processed_df, column='close'):
        """
        Plot a comparison of original vs. processed data
        
        Parameters:
        -----------
        original_df : pandas.DataFrame
            Original DataFrame
        processed_df : pandas.DataFrame
            Processed DataFrame
        column : str
            Column to plot
        """
        plt.figure(figsize=(14, 7))
        
        # Plot original data
        plt.subplot(2, 1, 1)
        plt.plot(original_df.index, original_df[column], label=f'Original {column}')
        plt.title(f'Original {column} data')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot processed data
        plt.subplot(2, 1, 2)
        diff_col = f'{column}_diff' if f'{column}_diff' in processed_df.columns else column
        plt.plot(processed_df.index, processed_df[diff_col], label=f'Processed {column}')
        plt.title(f'Processed {column} data (differenced and normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'data_preprocessing_{column}_comparison.png')
        plt.close()
        
    def plot_indicators(self, df):
        """
        Plot the technical indicators used in the model:
        - Relative Strength Index (RSI)
        - Normalized Average True Range (ATR)
        - Chaikin Money Flow (CMF)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with indicators
        """
        plt.figure(figsize=(15, 12))
        
        # Plot closing price
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(df.index, df['close'], label='Closing Price')
        ax1.set_title('Closing Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot RSI
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(df.index, df['rsi'], label='RSI', color='orange')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax2.set_title('Relative Strength Index (RSI)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot Normalized ATR
        if 'atr' in df.columns:
            ax3 = plt.subplot(4, 1, 3)
            ax3.plot(df.index, df['atr'], label='Normalized ATR', color='purple')
            ax3.set_title('Normalized Average True Range (ATR)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot CMF
        if 'cmf' in df.columns:
            ax4 = plt.subplot(4, 1, 4)
            ax4.plot(df.index, df['cmf'], label='CMF', color='blue')
            ax4.axhline(y=0.0, color='red', linestyle='--', alpha=0.5)
            ax4.set_title('Chaikin Money Flow (CMF)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('technical_indicators.png')
        plt.close()