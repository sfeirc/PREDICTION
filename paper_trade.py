"""
📈 PAPER TRADING MODE

Run the bot in paper trading mode to test live predictions WITHOUT risking real money.
Uses optimized models and real-time data from Binance.
"""

import sys
import io
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data_manager_worldclass import DataManagerWorldClass
from feature_engine_worldclass import FeatureEngineWorldClass
from lightgbm import LGBMClassifier


def load_best_params() -> dict:
    """Load optimized parameters"""
    best_path = Path('logs/best_settings.yaml')
    if not best_path.exists():
        logger.warning("No best_settings.yaml found, using defaults")
        return {
            'learning_rate': 0.03,
            'max_depth': 8,
            'num_leaves': 64,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    with open(best_path, 'r', encoding='utf-8') as f:
        best = yaml.safe_load(f) or {}
    
    return best.get('best_lightgbm_params', {
        'learning_rate': 0.03,
        'max_depth': 8,
        'num_leaves': 64,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    })


class PaperTradingBot:
    """Paper trading bot - test live without real money"""
    
    def __init__(self, config_path='config_ultimate.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}
        
        # Ensure correlation_pairs exists
        self.config.setdefault('data', {})
        self.config['data'].setdefault('correlation_pairs', ['ETHUSDT', 'BNBUSDT'])
        
        self.dm = DataManagerWorldClass(self.config)
        self.fe = FeatureEngineWorldClass(self.config)
        self.best_params = load_best_params()
        self.model = None
        self.positions = {}  # {symbol: {'side': 'LONG'/'SHORT', 'entry_price': float, 'size': float}}
        self.trades = []
        self.starting_balance = 10000.0
        self.balance = self.starting_balance
        self.confidence_threshold = 0.60  # Lower threshold for more trades (can adjust: 0.55-0.65)
        
    def train_model(self):
        """Train model on historical data"""
        logger.info("📊 Training model on historical data...")
        
        # Fetch data
        data = self.dm.fetch_all_data()
        df = self.fe.create_features(data)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Prepare features
        feature_cols = [c for c in df.columns if not c.startswith('target_')]
        target_cols = [c for c in df.columns if c.startswith('target_')]
        
        if not target_cols:
            logger.error("No target columns found!")
            return False
        
        target_col = target_cols[0]
        
        # Use last 80% for training
        split_idx = int(len(df) * 0.8)
        df_train = df.iloc[:split_idx]
        
        X_train = df_train[feature_cols].values
        y_train = df_train[target_col].values
        
        # Train
        self.model = LGBMClassifier(
            n_estimators=300,
            learning_rate=float(self.best_params.get('learning_rate', 0.03)),
            max_depth=int(self.best_params.get('max_depth', 8)),
            num_leaves=int(self.best_params.get('num_leaves', 64)),
            min_child_samples=int(self.best_params.get('min_child_samples', 20)),
            subsample=float(self.best_params.get('subsample', 0.8)),
            colsample_bytree=float(self.best_params.get('colsample_bytree', 0.8)),
            random_state=42,
            verbose=-1
        )
        
        self.model.fit(X_train, y_train)
        self.feature_cols = feature_cols
        logger.info(f"✅ Model trained on {len(df_train)} samples")
        return True
    
    def get_prediction(self, df_latest):
        """Get prediction for latest data with robust error handling and CONFIDENCE BOOST"""
        if self.model is None:
            logger.error("Model is None - cannot make prediction")
            return None
        
        # Check for missing features and fill with 0
        missing_features = [f for f in self.feature_cols if f not in df_latest.columns]
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features, filling with 0: {missing_features[:5]}...")
            for feat in missing_features:
                df_latest[feat] = 0.0
        
        # Extract features in correct order
        try:
            X = df_latest[self.feature_cols].fillna(0.0).values
        except KeyError as e:
            logger.error(f"KeyError extracting features: {e}")
            logger.error(f"Available columns: {list(df_latest.columns)[:20]}...")
            return None
        
        if X.shape[1] != len(self.feature_cols):
            logger.error(f"Feature dimension mismatch: {X.shape[1]} vs {len(self.feature_cols)}")
            return None
        
        # Check for NaN/Inf
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("NaN/Inf detected in features, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            # Get prediction
            proba_array = self.model.predict_proba(X)
            if proba_array is None or len(proba_array) == 0:
                logger.error("Model returned None or empty prediction")
                return None
            
            proba_raw = float(proba_array[:, 1][0])
            
            # Validate prediction
            if np.isnan(proba_raw) or np.isinf(proba_raw):
                logger.warning(f"Invalid prediction value: {proba_raw}, returning 0.5")
                return 0.5
            
            if proba_raw < 0 or proba_raw > 1:
                logger.warning(f"Prediction out of range: {proba_raw}, clipping to [0, 1]")
                proba_raw = max(0.0, min(1.0, proba_raw))
            
            # ===== CONFIDENCE BOOST: Push probabilities toward extremes (closer to 0 or 1) =====
            # Use sigmoid-like transformation to make predictions more confident
            # This pushes probabilities away from 0.5 toward 0 or 1
            
            # Method 1: Power transformation (stronger push toward extremes)
            # If proba > 0.5, push toward 1.0; if proba < 0.5, push toward 0.0
            boost_factor = 2.5  # Higher = more aggressive push toward extremes
            
            if proba_raw > 0.5:
                # Push toward 1.0
                # Transform: map [0.5, 1.0] to [0.5, 1.0] with more extreme values
                normalized = (proba_raw - 0.5) / 0.5  # [0, 1]
                boosted = normalized ** (1 / boost_factor)  # Push toward 1
                proba = 0.5 + boosted * 0.5
            else:
                # Push toward 0.0
                normalized = (0.5 - proba_raw) / 0.5  # [0, 1]
                boosted = normalized ** (1 / boost_factor)  # Push toward 1
                proba = 0.5 - boosted * 0.5
            
            # Ensure we stay in [0, 1] range
            proba = max(0.0, min(1.0, proba))
            
            # Log boost effect
            if abs(proba - proba_raw) > 0.05:
                logger.debug(f"Confidence boost: {proba_raw:.4f} → {proba:.4f} (boosted by {abs(proba - proba_raw):.4f})")
            
            return proba
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}", exc_info=True)
            return None
    
    def execute_trade(self, symbol, side, price, confidence):
        """Execute a paper trade - BALANCE COMPOUNDS (starts at $10k, grows with profits)"""
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'price': price,
            'confidence': confidence,
            'balance_before': self.balance
        }
        
        if side == 'BUY':
            # Use 2% of CURRENT balance (compounds!)
            position_size = self.balance * 0.02
            size = position_size / price
            
            # Reserve capital for this position
            self.balance -= position_size
            
            trade['size'] = size
            trade['position_size'] = position_size
            
            self.positions[symbol] = {
                'side': 'LONG',
                'entry_price': price,
                'size': size,
                'position_size': position_size,  # Store original investment
                'entry_time': datetime.now()
            }
            
        elif side == 'SELL' and symbol in self.positions:
            # Close existing position
            pos = self.positions[symbol]
            
            # Calculate P&L
            pnl_pct = (price - pos['entry_price']) / pos['entry_price']
            pnl = pos['position_size'] * pnl_pct
            exit_value = pos['position_size'] + pnl  # Original + profit
            
            # Add back position value + profit to balance (compounds!)
            self.balance += exit_value
            
            trade['size'] = pos['size']
            trade['entry_price'] = pos['entry_price']
            trade['exit_price'] = price
            trade['pnl'] = pnl
            trade['pnl_pct'] = pnl_pct
            trade['position_size'] = pos['position_size']
            trade['exit_value'] = exit_value
            
            del self.positions[symbol]
        
        trade['balance_after'] = self.balance
        self.trades.append(trade)
        return trade
    
    def run(self, duration_minutes=60, check_interval_seconds=60):
        """Run paper trading for specified duration"""
        logger.info("="*80)
        logger.info("📈 STARTING PAPER TRADING")
        logger.info("="*80)
        logger.info(f"Duration: {duration_minutes} minutes")
        logger.info(f"Check interval: {check_interval_seconds} seconds")
        logger.info(f"Starting balance: ${self.balance:,.2f}")
        logger.info(f"💰 COMPOUNDING MODE: Balance grows with each profitable trade!")
        logger.info(f"   Position size: 2% of CURRENT balance (not fixed)")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info("="*80)
        
        if not self.train_model():
            logger.error("Failed to train model!")
            return
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        iteration = 0
        
        try:
            while datetime.now() < end_time:
                iteration += 1
                logger.info(f"\n--- Iteration {iteration} ---")
                
                # Fetch latest data
                try:
                    data = self.dm.fetch_all_data()
                    df = self.fe.create_features(data)
                    df = df.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(df) == 0:
                        logger.warning("No data available")
                        time.sleep(check_interval_seconds)
                        continue
                    
                    # Get latest row
                    df_latest = df.iloc[[-1]]
                    
                    # Get prediction
                    proba = self.get_prediction(df_latest)
                    if proba is None:
                        logger.warning("Could not generate prediction - skipping this iteration")
                        time.sleep(check_interval_seconds)
                        continue
                    
                    # Get current price
                    current_price = df_latest['close'].iloc[0] if 'close' in df_latest.columns else None
                    symbol = self.config['data'].get('primary_pair', 'BTCUSDT')
                    
                    # Calculate confidence score (distance from 0.5)
                    confidence_score = abs(proba - 0.5) * 2  # Normalize to [0, 1]
                    direction = "UP" if proba > 0.5 else "DOWN"
                    
                    logger.info(f"📊 Prediction: {proba:.4f} ({direction}, confidence: {confidence_score:.2%})")
                    logger.info(f"💰 Current price: ${current_price:,.2f}" if current_price else "Price: N/A")
                    logger.info(f"🎯 Threshold: {self.confidence_threshold} (need proba > {self.confidence_threshold} for BUY or < {1-self.confidence_threshold} for SELL)")
                    
                    # Generate signal
                    buy_signal = proba > self.confidence_threshold
                    sell_signal = proba < (1 - self.confidence_threshold)
                    
                    if buy_signal:
                        # Strong buy signal
                        if symbol not in self.positions:
                            trade = self.execute_trade(symbol, 'BUY', current_price, proba)
                            logger.info(f"🟢 BUY EXECUTED - Entry: ${current_price:,.2f}, Confidence: {proba:.4f}")
                        else:
                            logger.info(f"⏸️  Already have LONG position, skipping BUY")
                    elif sell_signal:
                        # Strong sell signal
                        if symbol in self.positions and self.positions[symbol]['side'] == 'LONG':
                            trade = self.execute_trade(symbol, 'SELL', current_price, proba)
                            logger.info(f"🔴 SELL EXECUTED - Exit: ${current_price:,.2f}, Confidence: {proba:.4f}")
                        else:
                            logger.info(f"⏸️  No open LONG position to close")
                    else:
                        logger.info(f"⏸️  HOLD - Confidence {confidence_score:.2%} below threshold ({self.confidence_threshold:.2%})")
                    
                    # Status
                    pnl = self.balance - self.starting_balance
                    pnl_pct = (pnl / self.starting_balance) * 100
                    logger.info(f"Balance: ${self.balance:,.2f} ({pnl:+,.2f}, {pnl_pct:+.2f}%)")
                    logger.info(f"Open positions: {len(self.positions)}")
                    logger.info(f"Total trades: {len(self.trades)}")
                    
                except Exception as e:
                    logger.error(f"Error in iteration {iteration}: {e}", exc_info=True)
                
                # Sleep
                time.sleep(check_interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("\n⚠️  Stopped by user")
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("📊 PAPER TRADING SUMMARY")
        logger.info("="*80)
        
        final_pnl = self.balance - self.starting_balance
        final_pnl_pct = (final_pnl / self.starting_balance) * 100
        
        logger.info(f"Starting balance: ${self.starting_balance:,.2f}")
        logger.info(f"Final balance: ${self.balance:,.2f}")
        logger.info(f"Total PnL: ${final_pnl:+,.2f} ({final_pnl_pct:+.2f}%)")
        logger.info(f"Total trades: {len(self.trades)}")
        logger.info(f"Open positions: {len(self.positions)}")
        
        # Win rate
        closed_trades = [t for t in self.trades if 'pnl' in t]
        if closed_trades:
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            win_rate = len(winning_trades) / len(closed_trades)
            logger.info(f"Win rate: {win_rate:.2%} ({len(winning_trades)}/{len(closed_trades)})")
        
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_path = Path('logs/paper_trades.csv')
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"\n✅ Trades saved to {trades_path}")
        
        logger.info("="*80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper Trading Bot")
    parser.add_argument('--duration', type=int, default=60, help='Duration in minutes')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    
    args = parser.parse_args()
    
    bot = PaperTradingBot()
    bot.run(duration_minutes=args.duration, check_interval_seconds=args.interval)


if __name__ == '__main__':
    main()

