"""
🏆 WORLD-CLASS CRYPTO TRADING BOT 🏆

Inspired by: Freqtrade, Jesse, QuantConnect, Catalyst
Features:
- Multi-timeframe analysis
- Real order book integration
- Ensemble ML models
- Professional risk management
- Walk-forward optimization
- Live trading capability
- Real-time monitoring
"""

import sys
import io
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/worldclass_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class WorldClassTradingBot:
    """
    Professional-grade trading bot with enterprise features.
    """
    
    def __init__(self, config_path: str = "config_worldclass.yaml"):
        """Initialize the trading bot."""
        logger.info("=" * 80)
        logger.info("🏆 WORLD-CLASS TRADING BOT INITIALIZING 🏆")
        logger.info("=" * 80)
        
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_manager = None
        self.feature_engine = None
        self.model_ensemble = None
        self.risk_manager = None
        self.executor = None
        self.monitor = None
        
        # State
        self.is_running = False
        self.positions = {}
        self.orders = {}
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'peak_capital': self.config['risk']['initial_capital'],
            'max_drawdown': 0.0,
        }
        
        logger.info("✅ Bot initialized successfully")
    
    def setup(self):
        """Setup all bot components."""
        logger.info("\n🔧 Setting up bot components...")
        
        # 1. Data Manager (multi-timeframe, order book)
        from data_manager_worldclass import DataManagerWorldClass
        self.data_manager = DataManagerWorldClass(self.config)
        logger.info("✅ Data Manager ready")
        
        # 2. Feature Engine (multi-timeframe features)
        from feature_engine_worldclass import FeatureEngineWorldClass
        self.feature_engine = FeatureEngineWorldClass(self.config)
        logger.info("✅ Feature Engine ready")
        
        # 3. Model Ensemble (LightGBM + XGBoost + CatBoost + LSTM)
        from model_ensemble_worldclass import ModelEnsembleWorldClass
        self.model_ensemble = ModelEnsembleWorldClass(self.config)
        logger.info("✅ Model Ensemble ready")
        
        # 4. Risk Manager (Kelly, portfolio heat, drawdown limits)
        from risk_manager_worldclass import RiskManagerWorldClass
        self.risk_manager = RiskManagerWorldClass(self.config)
        logger.info("✅ Risk Manager ready")
        
        # 5. Executor (live trading engine)
        from executor_worldclass import ExecutorWorldClass
        self.executor = ExecutorWorldClass(self.config)
        logger.info("✅ Executor ready")
        
        # 6. Monitor (dashboards, alerts)
        from monitor_worldclass import MonitorWorldClass
        self.monitor = MonitorWorldClass(self.config)
        logger.info("✅ Monitor ready")
        
        logger.info("\n🎉 All components ready!")
    
    def train(self, mode: str = "walk_forward"):
        """
        Train the ensemble model.
        
        Args:
            mode: "walk_forward" or "standard"
        """
        logger.info("\n" + "=" * 80)
        logger.info("🎓 TRAINING MODELS")
        logger.info("=" * 80)
        
        # Fetch historical data
        logger.info("\n📥 Fetching multi-timeframe data...")
        data = self.data_manager.fetch_all_data()
        
        # Create features
        logger.info("\n🔧 Engineering features...")
        features_df = self.feature_engine.create_features(data)
        
        # Train ensemble
        if mode == "walk_forward":
            logger.info("\n🚶 Walk-forward optimization...")
            results = self.model_ensemble.train_walk_forward(features_df, self.config)
        else:
            logger.info("\n🎯 Standard training...")
            results = self.model_ensemble.train(features_df, self.config)
        
        logger.info("\n✅ Training complete!")
        return results
    
    def backtest(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Run realistic backtest.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 BACKTESTING")
        logger.info("=" * 80)
        
        from backtest_engine_worldclass import BacktestEngineWorldClass
        backtest_engine = BacktestEngineWorldClass(self.config)
        
        results = backtest_engine.run(
            data_manager=self.data_manager,
            feature_engine=self.feature_engine,
            model_ensemble=self.model_ensemble,
            risk_manager=self.risk_manager,
            start_date=start_date,
            end_date=end_date,
        )
        
        # Generate report
        self.monitor.generate_backtest_report(results)
        
        return results
    
    def run_live(self):
        """Run the bot in live trading mode."""
        logger.info("\n" + "=" * 80)
        logger.info("🚀 STARTING LIVE TRADING")
        logger.info("=" * 80)
        
        if self.config['execution']['testnet']:
            logger.warning("⚠️  Running on TESTNET (paper trading)")
        else:
            logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!")
            confirm = input("Type 'CONFIRM' to proceed: ")
            if confirm != "CONFIRM":
                logger.info("Cancelled by user")
                return
        
        self.is_running = True
        
        try:
            while self.is_running:
                # 1. Update market data
                market_data = self.data_manager.get_latest_data()
                
                # 2. Check risk limits
                if not self.risk_manager.check_limits(self.performance):
                    logger.warning("⚠️  Risk limits breached - pausing trading")
                    self.monitor.send_alert("Risk limits breached", level="warning")
                    break
                
                # 3. Generate features
                features = self.feature_engine.create_live_features(market_data)
                
                # 4. Get predictions
                predictions = self.model_ensemble.predict(features)
                
                # 5. Generate signals
                signals = self.generate_signals(predictions, market_data)
                
                # 6. Execute trades
                for signal in signals:
                    if self.risk_manager.approve_trade(signal, self.positions):
                        self.executor.execute_trade(signal)
                
                # 7. Update positions
                self.update_positions()
                
                # 8. Update monitoring
                self.monitor.update_dashboard(self.positions, self.performance)
                
                # Sleep until next iteration
                self.wait_for_next_candle()
                
        except KeyboardInterrupt:
            logger.info("\n⚠️  Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error in live trading: {e}")
            self.monitor.send_alert(f"Critical error: {e}", level="critical")
        finally:
            self.shutdown()
    
    def generate_signals(self, predictions: Dict, market_data: Dict) -> List[Dict]:
        """Generate trading signals from predictions."""
        signals = []
        
        for symbol, pred in predictions.items():
            # Check confidence
            if pred['confidence'] < self.config['risk']['min_confidence']:
                continue
            
            # Check market conditions
            if not self.is_favorable_market(symbol, market_data):
                continue
            
            # Create signal
            signal = {
                'symbol': symbol,
                'side': 'BUY' if pred['direction'] == 1 else 'SELL',
                'confidence': pred['confidence'],
                'target': pred['target_price'],
                'stop_loss': pred['stop_loss'],
                'take_profit': pred['take_profit'],
                'timestamp': datetime.now(),
            }
            
            signals.append(signal)
        
        return signals
    
    def is_favorable_market(self, symbol: str, market_data: Dict) -> bool:
        """Check if market conditions are favorable for trading."""
        data = market_data[symbol]
        
        # Check liquidity
        if data['volume_24h'] < self.config['risk']['min_liquidity']:
            return False
        
        # Check volatility
        if data['volatility_1h'] > self.config['risk']['max_volatility']:
            return False
        
        # Check spread
        if data['spread'] > 0.001:  # 0.1%
            return False
        
        return True
    
    def update_positions(self):
        """Update open positions and check exit conditions."""
        for position_id, position in list(self.positions.items()):
            # Get current price
            current_price = self.data_manager.get_current_price(position['symbol'])
            
            # Calculate PnL
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # Stop loss
            if pnl_pct <= -self.config['risk']['stop_loss']:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Take profit
            elif pnl_pct >= self.config['risk']['take_profit']:
                should_exit = True
                exit_reason = "take_profit"
            
            # Trailing stop
            elif position.get('peak_pnl', 0) - pnl_pct >= self.config['risk']['trailing_stop']:
                should_exit = True
                exit_reason = "trailing_stop"
            
            # Time-based exit (if position is open for too long)
            elif (datetime.now() - position['entry_time']).seconds > 14400:  # 4 hours
                should_exit = True
                exit_reason = "timeout"
            
            if should_exit:
                self.executor.close_position(position_id, exit_reason)
                del self.positions[position_id]
                logger.info(f"Position closed: {position['symbol']} | Reason: {exit_reason} | PnL: {pnl_pct:.2%}")
            else:
                # Update peak PnL for trailing stop
                position['peak_pnl'] = max(position.get('peak_pnl', 0), pnl_pct)
    
    def wait_for_next_candle(self):
        """Wait until the next candle opens."""
        import time
        
        now = datetime.now()
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        sleep_seconds = (next_minute - now).total_seconds()
        
        time.sleep(max(0, sleep_seconds))
    
    def shutdown(self):
        """Gracefully shutdown the bot."""
        logger.info("\n" + "=" * 80)
        logger.info("🛑 SHUTTING DOWN BOT")
        logger.info("=" * 80)
        
        self.is_running = False
        
        # Close all positions
        logger.info("Closing all open positions...")
        for position_id in list(self.positions.keys()):
            self.executor.close_position(position_id, "shutdown")
        
        # Cancel all orders
        logger.info("Cancelling all open orders...")
        for order_id in list(self.orders.keys()):
            self.executor.cancel_order(order_id)
        
        # Save performance report
        logger.info("Generating final report...")
        self.monitor.generate_final_report(self.performance)
        
        logger.info("\n✅ Bot shutdown complete")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="World-Class Crypto Trading Bot")
    parser.add_argument("--mode", choices=["train", "backtest", "live"], required=True,
                       help="Bot mode: train, backtest, or live")
    parser.add_argument("--config", default="config_worldclass.yaml",
                       help="Path to config file")
    parser.add_argument("--start-date", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Backtest end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = WorldClassTradingBot(config_path=args.config)
    bot.setup()
    
    # Run in specified mode
    if args.mode == "train":
        bot.train(mode="walk_forward")
    
    elif args.mode == "backtest":
        bot.backtest(start_date=args.start_date, end_date=args.end_date)
    
    elif args.mode == "live":
        bot.run_live()


if __name__ == "__main__":
    main()

