"""
Binance data fetcher with caching for 1-minute OHLCV data.
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import requests
from tqdm import tqdm


class BinanceDataFetcher:
    """Fetch 1-minute OHLCV data from Binance public API with caching."""

    BASE_URL = "https://api.binance.com/api/v3/klines"
    DEPTH_URL = "https://api.binance.com/api/v3/depth"

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> list:
        """
        Fetch klines from Binance.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of klines to fetch (max 1000)

        Returns:
            List of klines
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return []

    def fetch_historical_data(
        self,
        symbol: str,
        days: int = 30,
        interval: str = "1m",
    ) -> pd.DataFrame:
        """
        Fetch historical data for specified number of days.

        Args:
            symbol: Trading pair
            days: Number of days to fetch
            interval: Kline interval

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching {days} days of {interval} data for {symbol}...")

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        all_klines = []
        current_start = start_ms

        # Calculate total iterations for progress bar
        interval_ms = self._interval_to_ms(interval)
        total_klines = (end_ms - start_ms) // interval_ms
        iterations = (total_klines // 1000) + 1

        pbar = tqdm(total=iterations, desc=f"Fetching {symbol}")

        while current_start < end_ms:
            klines = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_ms,
                limit=1000,
            )

            if not klines:
                break

            all_klines.extend(klines)

            # Move to next batch
            # Binance returns klines with open time, use the last open time + interval
            current_start = klines[-1][0] + interval_ms

            pbar.update(1)

        pbar.close()

        print(f"Fetched {len(all_klines)} klines")

        # Convert to DataFrame
        df = self._klines_to_dataframe(all_klines)

        return df

    def _interval_to_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds."""
        unit = interval[-1]
        value = int(interval[:-1])

        ms_map = {
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
        }

        return value * ms_map[unit]

    def _klines_to_dataframe(self, klines: list) -> pd.DataFrame:
        """
        Convert klines list to DataFrame.

        Kline structure:
        [
            [
                1499040000000,      // 0: Open time
                "0.01634000",       // 1: Open
                "0.80000000",       // 2: High
                "0.01575800",       // 3: Low
                "0.01577100",       // 4: Close
                "148976.11427815",  // 5: Volume
                1499644799999,      // 6: Close time
                "2434.19055334",    // 7: Quote asset volume
                308,                // 8: Number of trades
                "1756.87402397",    // 9: Taker buy base asset volume
                "28.46694368",      // 10: Taker buy quote asset volume
                "17928899.62484339" // 11: Ignore
            ]
        ]
        """
        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        # Convert types
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
            df[col] = df[col].astype(float)

        df["trades"] = df["trades"].astype(int)

        # Set index
        df.set_index("open_time", inplace=True)

        # Drop unnecessary columns
        df.drop(columns=["close_time", "ignore"], inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        # Sort by time
        df.sort_index(inplace=True)

        return df

    def cache_path(self, symbol: str, interval: str = "1m") -> Path:
        """Get cache file path for symbol."""
        return self.cache_dir / f"{symbol.lower()}_{interval}.parquet"

    def load_from_cache(self, symbol: str, interval: str = "1m") -> Optional[pd.DataFrame]:
        """Load data from cache if exists."""
        cache_file = self.cache_path(symbol, interval)

        if cache_file.exists():
            print(f"Loading {symbol} from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            print(f"Loaded {len(df)} rows from cache")
            return df

        return None

    def save_to_cache(self, df: pd.DataFrame, symbol: str, interval: str = "1m"):
        """Save DataFrame to cache."""
        cache_file = self.cache_path(symbol, interval)
        df.to_parquet(cache_file, compression="snappy")
        print(f"Saved to cache: {cache_file}")

    def fetch_orderbook(self, symbol: str, limit: int = 5) -> Dict:
        """
        Fetch order book depth from Binance.

        Args:
            symbol: Trading pair
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            Dictionary with bids and asks
        """
        params = {
            "symbol": symbol,
            "limit": limit,
        }

        try:
            response = requests.get(self.DEPTH_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "bids": [[float(price), float(qty)] for price, qty in data.get("bids", [])],
                "asks": [[float(price), float(qty)] for price, qty in data.get("asks", [])],
                "timestamp": pd.Timestamp.now(),
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching order book: {e}")
            return {"bids": [], "asks": [], "timestamp": pd.Timestamp.now()}

    def fetch_and_cache(
        self,
        symbol: str,
        days: int = 30,
        interval: str = "1m",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch data and cache it. If cached and not refresh, load from cache.

        Args:
            symbol: Trading pair
            days: Number of days to fetch
            interval: Kline interval
            refresh: Force refresh from Binance

        Returns:
            DataFrame with OHLCV data
        """
        if not refresh:
            cached = self.load_from_cache(symbol, interval)
            if cached is not None:
                return cached

        # Fetch from Binance
        df = self.fetch_historical_data(symbol, days, interval)

        # Save to cache
        self.save_to_cache(df, symbol, interval)

        return df


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance data")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading pair (default: BTCUSDT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to fetch (default: 30)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="Kline interval (default: 1m)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh from Binance (ignore cache)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/raw",
        help="Cache directory (default: data/raw)",
    )

    args = parser.parse_args()

    fetcher = BinanceDataFetcher(cache_dir=args.cache_dir)
    df = fetcher.fetch_and_cache(
        symbol=args.symbol,
        days=args.days,
        interval=args.interval,
        refresh=args.refresh,
    )

    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Rows: {len(df)}")
    print(f"Start: {df.index.min()}")
    print(f"End: {df.index.max()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()

