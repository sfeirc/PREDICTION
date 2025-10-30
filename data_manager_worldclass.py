"""
Multi-Timeframe Data Manager with Real Order Book Integration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime, timedelta
import requests
import time

logger = logging.getLogger(__name__)


class DataManagerWorldClass:
    """
    Manages multi-timeframe data and real order book feeds.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = Path(config['data']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = "https://api.binance.com/api/v3"
        
        logger.info("📊 Data Manager initialized")
    
    def fetch_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch all required data (multi-timeframe, multi-asset).
        
        Returns:
            Dict of {symbol: {timeframe: DataFrame}}
        """
        logger.info("\n🔄 Fetching multi-timeframe data...")
        
        data = {}
        
        # Primary pair
        primary_pair = self.config['data']['primary_pair']
        data[primary_pair] = self._fetch_symbol_data(primary_pair)
        
        # Correlation pairs
        for pair in self.config['data']['correlation_pairs']:
            data[pair] = self._fetch_symbol_data(pair)
        
        logger.info(f"✅ Fetched data for {len(data)} symbols")
        
        return data
    
    def _fetch_symbol_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch multi-timeframe data for a symbol."""
        logger.info(f"   Fetching {symbol}...")
        
        timeframes_data = {}
        
        # Fetch primary timeframe (1m) first
        primary_tf = self.config['data']['timeframes']['primary']
        df = self._fetch_klines(symbol, primary_tf)
        timeframes_data[primary_tf] = df
        logger.info(f"      {primary_tf}: {len(df)} candles")
        
        # Fetch analysis timeframes
        for tf in self.config['data']['timeframes']['analysis']:
            if tf != primary_tf:  # Don't fetch twice
                df = self._fetch_klines(symbol, tf)
                timeframes_data[tf] = df
                logger.info(f"      {tf}: {len(df)} candles")
        
        return timeframes_data
    
    def _fetch_klines(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch klines from Binance with caching.
        """
        # Check cache
        cache_file = self.cache_dir / f"{symbol}_{interval}.parquet"
        
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            # Update with latest data
            last_time = df.index[-1]
            new_df = self._download_klines(symbol, interval, start_time=int(last_time.timestamp() * 1000))
            if len(new_df) > 0:
                df = pd.concat([df, new_df]).drop_duplicates()
        else:
            # Download all
            days = self.config['data']['history_days']
            df = self._download_klines(symbol, interval, days=days)
        
        # Save cache
        df.to_parquet(cache_file)
        
        return df
    
    def _download_klines(self, symbol: str, interval: str, days: int = None, start_time: int = None) -> pd.DataFrame:
        """Download klines from Binance API."""
        url = f"{self.base_url}/klines"
        
        all_data = []
        limit = 1000
        
        if days:
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        current_start = start_time
        
        while True:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit,
            }
            
            if current_start:
                params['startTime'] = current_start
            
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # Update start time for next batch
                current_start = data[-1][6] + 1  # Close time + 1ms
                
                # Check if we got less than limit (end of data)
                if len(data) < limit:
                    break
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching klines: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('open_time')
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                       'trades', 'taker_buy_base', 'taker_buy_quote']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        df = df.drop(['close_time', 'ignore'], axis=1)
        
        return df
    
    def fetch_orderbook(self, symbol: str, depth: int = 20) -> Dict:
        """
        Fetch real-time order book data.
        
        Returns:
            {
                'bids': [[price, quantity], ...],
                'asks': [[price, quantity], ...],
                'timestamp': int
            }
        """
        url = f"{self.base_url}/depth"
        params = {
            'symbol': symbol,
            'limit': depth
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            return {
                'bids': [[float(p), float(q)] for p, q in data['bids']],
                'asks': [[float(p), float(q)] for p, q in data['asks']],
                'timestamp': datetime.now().timestamp() * 1000,
            }
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None
    
    def calculate_orderbook_features(self, orderbook: Dict) -> Dict:
        """
        Calculate advanced order book features.
        """
        if not orderbook:
            return {}
        
        bids = np.array(orderbook['bids'])
        asks = np.array(orderbook['asks'])
        
        # Best bid/ask
        best_bid = bids[0, 0]
        best_ask = asks[0, 0]
        
        # Spread
        spread = (best_ask - best_bid) / best_bid
        
        # Mid price
        mid_price = (best_bid + best_ask) / 2
        
        # Microprice (volume-weighted)
        bid_vol = bids[0, 1]
        ask_vol = asks[0, 1]
        microprice = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
        
        # Order imbalance
        total_bid_vol = bids[:, 1].sum()
        total_ask_vol = asks[:, 1].sum()
        order_imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        
        # Book pressure (first 5 levels)
        bid_pressure = (bids[:5, 0] * bids[:5, 1]).sum()
        ask_pressure = (asks[:5, 0] * asks[:5, 1]).sum()
        book_pressure = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure)
        
        # Depth imbalance (volume at different price levels)
        bid_depth = bids[:, 1].sum()
        ask_depth = asks[:, 1].sum()
        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        
        # VWAP (volume-weighted average price)
        vwap_bid = (bids[:, 0] * bids[:, 1]).sum() / bid_depth if bid_depth > 0 else best_bid
        vwap_ask = (asks[:, 0] * asks[:, 1]).sum() / ask_depth if ask_depth > 0 else best_ask
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid_price,
            'spread': spread,
            'microprice': microprice,
            'order_imbalance': order_imbalance,
            'book_pressure': book_pressure,
            'depth_imbalance': depth_imbalance,
            'bid_vol': bid_vol,
            'ask_vol': ask_vol,
            'total_bid_vol': total_bid_vol,
            'total_ask_vol': total_ask_vol,
            'vwap_bid': vwap_bid,
            'vwap_ask': vwap_ask,
        }
    
    def get_latest_data(self) -> Dict:
        """Get latest market data for all symbols."""
        data = {}
        
        for symbol in [self.config['data']['primary_pair']] + self.config['data']['correlation_pairs']:
            # Get latest candle
            df = self._fetch_klines(symbol, "1m", limit=1)
            
            # Get order book
            orderbook = self.fetch_orderbook(symbol)
            orderbook_features = self.calculate_orderbook_features(orderbook)
            
            # Get 24h stats
            stats = self._get_24h_stats(symbol)
            
            data[symbol] = {
                'price': df['close'].iloc[-1],
                'volume_24h': stats['volume'],
                'volatility_1h': self._calculate_volatility(symbol, '1h'),
                'spread': orderbook_features.get('spread', 0),
                'orderbook': orderbook_features,
            }
        
        return data
    
    def _get_24h_stats(self, symbol: str) -> Dict:
        """Get 24h ticker statistics."""
        url = f"{self.base_url}/ticker/24hr"
        params = {'symbol': symbol}
        
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            return {
                'volume': float(data['quoteVolume']),
                'price_change_pct': float(data['priceChangePercent']),
            }
        except Exception as e:
            logger.error(f"Error fetching 24h stats: {e}")
            return {'volume': 0, 'price_change_pct': 0}
    
    def _calculate_volatility(self, symbol: str, period: str = '1h') -> float:
        """Calculate recent volatility."""
        df = self._fetch_klines(symbol, "1m", limit=60)
        returns = df['close'].pct_change().dropna()
        return returns.std() * np.sqrt(60)  # Annualized
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        url = f"{self.base_url}/ticker/price"
        params = {'symbol': symbol}
        
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data['price'])
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            return 0.0

