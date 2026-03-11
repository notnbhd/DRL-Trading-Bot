import os
import pandas as pd
import numpy as np
import joblib
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class CachedDataProcessor:
    """Process cryptocurrency data for the trading bot with caching support"""
    
    def __init__(self, cache_dir='data_cache'):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.cache_dir = cache_dir
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def download_data(self, symbol, interval, start_str, end_str=None, source='binance', use_cache=True):
        """
        Download historical data from the specified source or load from cache if available
        
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
        use_cache : bool, optional
            Whether to use cached data if available (default: True)
            
        Returns:
        --------
        DataFrame containing the historical data
        """
        # Create a cache filename
        end_date_str = end_str if end_str else 'latest'
        cache_file = f"{self.cache_dir}/{symbol}_{interval}_{start_str}_to_{end_date_str}.csv"
        
        # Check if cache file exists and use it if requested
        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file)
            
            # Convert timestamp back to datetime index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Convert price and volume columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            return df
        
        print(f"Downloading data from {source}...")
        if source == 'binance':
            from binance.client import Client
            # Use None, None for public API access without auth keys
            client = Client(None, None)  
            
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
            
            # Convert price and volume columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Keep only essential columns (with timestamp for saving to CSV)
            df = df[['timestamp'] + numeric_columns]
            
            # Save to cache file
            print(f"Saving data to cache: {cache_file}")
            df.to_csv(cache_file, index=False)
            
            # Set index after saving to CSV
            df.set_index('timestamp', inplace=True)
            
            return df
        else:
            raise ValueError(f"Data source '{source}' not supported")
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the DataFrame as specified in Table 1:
        1. Relative Strength Index (RSI)
        2. Normalized Average True Range (ATR)
        3. On-Balance Volume (OBV)
        
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
        df_processed['norm_atr'] = atr_values / df_processed['close']
        
        # 3. OBV (On-Balance Volume) - "On-balance volume indicator" in Table 1
        obv_indicator = OnBalanceVolumeIndicator(close=df_processed['close'], volume=df_processed['volume'])
        df_processed['obv'] = obv_indicator.on_balance_volume()
        
        # Normalize OBV for better scale compatibility
        df_processed['norm_obv'] = df_processed['obv'] / df_processed['obv'].abs().max()
        
        # Drop the raw OBV column
        df_processed.drop('obv', axis=1, inplace=True)
        
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
        
        # Apply first-order differencing to price columns to remove trend
        for col in ['open', 'high', 'low', 'close']:
            df_diff[f'{col}_diff'] = df_diff[col].diff()
            # Keep original columns as well
            df_diff[col] = df_diff[col]
        
        # Apply differencing to volume as well to ensure stationarity
        df_diff['volume_diff'] = df_diff['volume'].diff()
        
        # For indicators, check if they need differencing based on stationarity
        indicators = ['rsi', 'norm_atr', 'norm_obv']
        for indicator in indicators:
            if indicator in df_diff.columns:
                # RSI is already stationary by design, don't difference
                if indicator != 'rsi':
                    df_diff[f'{indicator}_diff'] = df_diff[indicator].diff()
        
        # Drop the first row with NaN values from differencing
        df_diff.dropna(inplace=True)
        
        return df_diff
    
    def fit_normalize(self, df):
        """
        Fit the scaler on data and transform it. Use ONLY on training data.
        """
        df_normalized = df.copy()
        preserve_columns = ['close_orig']
        normalized_columns = [col for col in df_normalized.columns if col not in preserve_columns]
        normalized_data = self.scaler.fit_transform(df_normalized[normalized_columns])
        normalized_df = pd.DataFrame(normalized_data, columns=normalized_columns, index=df_normalized.index)
        for col in preserve_columns:
            if col in df_normalized.columns:
                normalized_df[col] = df_normalized[col]
        return normalized_df
    
    def transform_normalize(self, df):
        """
        Transform data using an already-fitted scaler. Use for test/backtest/live data.
        """
        df_normalized = df.copy()
        preserve_columns = ['close_orig']
        normalized_columns = [col for col in df_normalized.columns if col not in preserve_columns]
        normalized_data = self.scaler.transform(df_normalized[normalized_columns])
        normalized_df = pd.DataFrame(normalized_data, columns=normalized_columns, index=df_normalized.index)
        for col in preserve_columns:
            if col in df_normalized.columns:
                normalized_df[col] = df_normalized[col]
        return normalized_df
    
    def save_scaler(self, path):
        """Save the fitted scaler to disk"""
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")
    
    def load_scaler(self, path):
        """Load a previously fitted scaler from disk"""
        self.scaler = joblib.load(path)
        print(f"Scaler loaded from {path}")
    
    def prepare_data(self, df, add_indicators=True, apply_diff=True, normalize=True, fit_scaler=True):
        """
        Prepare data for training by applying all preprocessing steps.
        
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
        fit_scaler : bool, optional
            If True, fit the scaler on this data (training data only).
            If False, use an already-fitted scaler to transform.
        """
        processed_df = df.copy()
        
        if add_indicators:
            processed_df = self.add_technical_indicators(processed_df)
        
        if apply_diff:
            processed_df = self.apply_difference(processed_df)
        
        if normalize:
            if fit_scaler:
                processed_df = self.fit_normalize(processed_df)
            else:
                processed_df = self.transform_normalize(processed_df)
        
        print(f"Prepared data with features: {processed_df.columns.tolist()}")
        print(f"Data includes lookback window of {100} hours with closing price and technical indicators")
        
        return processed_df
    
    def cache_processed_data(self, df, symbol, start_str, end_str=None, suffix='processed'):
        """
        Save processed data to cache
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Processed DataFrame to save
        symbol : str
            The trading pair symbol
        start_str : str
            Start date in format 'YYYY-MM-DD'
        end_str : str, optional
            End date in format 'YYYY-MM-DD'
        suffix : str, optional
            Suffix to add to filename
        """
        end_date_str = end_str if end_str else 'latest'
        cache_file = f"{self.cache_dir}/{symbol}_{start_str}_to_{end_date_str}_{suffix}.csv"
        
        # Reset index to save timestamp as a column
        df_to_save = df.reset_index()
        
        print(f"Saving processed data to: {cache_file}")
        df_to_save.to_csv(cache_file, index=False)
    
    def load_processed_data(self, symbol, start_str, end_str=None, suffix='processed'):
        """
        Load processed data from cache
        
        Parameters:
        -----------
        symbol : str
            The trading pair symbol
        start_str : str
            Start date in format 'YYYY-MM-DD'
        end_str : str, optional
            End date in format 'YYYY-MM-DD'
        suffix : str, optional
            Suffix in the filename
            
        Returns:
        --------
        Processed DataFrame or None if not found
        """
        end_date_str = end_str if end_str else 'latest'
        cache_file = f"{self.cache_dir}/{symbol}_{start_str}_to_{end_date_str}_{suffix}.csv"
        
        if os.path.exists(cache_file):
            print(f"Loading processed data from: {cache_file}")
            df = pd.read_csv(cache_file)
            
            # Convert timestamp back to datetime index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
        else:
            print(f"Processed data file not found: {cache_file}")
            return None
    
    def get_data(self, symbol, interval, start_str, end_str=None, use_cache=True, 
                 use_processed_cache=True, save_processed=True):
        """
        Get data for training - handles both downloading raw data and loading/saving processed data
        
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
        use_cache : bool, optional
            Whether to use cached raw data (default: True)
        use_processed_cache : bool, optional
            Whether to use cached processed data (default: True)
        save_processed : bool, optional
            Whether to save processed data (default: True)
            
        Returns:
        --------
        Processed DataFrame ready for training
        """
        # Try to load processed data first
        if use_processed_cache:
            processed_df = self.load_processed_data(symbol, start_str, end_str)
            if processed_df is not None:
                return processed_df
        
        # If no processed data available, download/load raw data and process it
        raw_df = self.download_data(symbol, interval, start_str, end_str, use_cache=use_cache)
        processed_df = self.prepare_data(raw_df)
        
        # Save processed data if requested
        if save_processed:
            self.cache_processed_data(processed_df, symbol, start_str, end_str)
        
        return processed_df 