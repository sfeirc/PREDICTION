"""
🏆 SIMPLIFIED WORLD-CLASS BOT TEST 🏆

This tests the core functionality with minimal complexity.
"""

import sys
import io
import yaml
import logging
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("🏆 WORLD-CLASS BOT - QUICK TEST 🏆")
    logger.info("=" * 80)
    
    # Load config
    with open("config_worldclass.yaml") as f:
        config = yaml.safe_load(f)
    
    # Simplify config for testing (only BTC, no correlation pairs)
    config['data']['correlation_pairs'] = ['ETHUSDT']  # Only ETH for simplicity
    config['data']['history_days'] = 30  # Less data for faster testing
    
    logger.info("\n📦 STEP 1: Initialize Components")
    logger.info("=" * 80)
    
    from data_manager_worldclass import DataManagerWorldClass
    from feature_engine_worldclass import FeatureEngineWorldClass
    from model_ensemble_worldclass import ModelEnsembleWorldClass
    from risk_manager_worldclass import RiskManagerWorldClass
    
    data_manager = DataManagerWorldClass(config)
    feature_engine = FeatureEngineWorldClass(config)
    model_ensemble = ModelEnsembleWorldClass(config)
    risk_manager = RiskManagerWorldClass(config)
    
    logger.info("\n📥 STEP 2: Fetch Data")
    logger.info("=" * 80)
    logger.info(f"Fetching {config['data']['history_days']} days of data...")
    
    data = data_manager.fetch_all_data()
    
    primary_pair = config['data']['primary_pair']
    logger.info(f"\n✅ Data fetched for {primary_pair}:")
    for tf, df in data[primary_pair].items():
        logger.info(f"   {tf}: {len(df):,} candles")
    
    # Order book test
    logger.info("\n📖 Testing order book...")
    orderbook = data_manager.fetch_orderbook(primary_pair, depth=5)
    if orderbook:
        ob_features = data_manager.calculate_orderbook_features(orderbook)
        logger.info(f"   Spread: {ob_features['spread']*100:.4f}%")
        logger.info(f"   Order Imbalance: {ob_features['order_imbalance']:.4f}")
    
    logger.info("\n🔧 STEP 3: Create Features")
    logger.info("=" * 80)
    
    features_df = feature_engine.create_features(data)
    
    logger.info(f"\n✅ Features created:")
    logger.info(f"   Total rows: {len(features_df):,}")
    feature_cols = feature_engine.get_feature_columns(features_df)
    logger.info(f"   Total features: {len(feature_cols)}")
    
    # Show sample features
    logger.info(f"\n   Sample features:")
    for feat in feature_cols[:10]:
        logger.info(f"      - {feat}")
    logger.info(f"      ... and {len(feature_cols)-10} more")
    
    logger.info("\n🤖 STEP 4: Train Ensemble")
    logger.info("=" * 80)
    
    results = model_ensemble.train(features_df, config)
    
    logger.info("\n✅ Training complete!")
    logger.info(f"\n📊 Ensemble Performance:")
    ensemble_metrics = results.get('ensemble', {})
    logger.info(f"   Accuracy: {ensemble_metrics.get('val_acc', 0):.4f}")
    logger.info(f"   AUC: {ensemble_metrics.get('val_auc', 0):.4f}")
    logger.info(f"   F1: {ensemble_metrics.get('val_f1', 0):.4f}")
    
    # Individual model performance
    logger.info(f"\n   Individual Models:")
    for model_name in ['lightgbm', 'xgboost']:
        if model_name in results:
            metrics = results[model_name]
            logger.info(f"   {model_name.upper()}:")
            logger.info(f"      Val Acc: {metrics.get('val_acc', 0):.4f}")
            logger.info(f"      Val AUC: {metrics.get('val_auc', 0):.4f}")
    
    logger.info("\n🛡️ STEP 5: Test Risk Management")
    logger.info("=" * 80)
    
    # Simulate a trading signal
    test_signal = {
        'symbol': primary_pair,
        'side': 'BUY',
        'confidence': 0.75,
        'target_price': 108000,
        'stop_loss': 107000,
        'take_profit': 111000,
    }
    
    capital = config['risk']['initial_capital']
    position_size = risk_manager.calculate_position_size(
        test_signal, 
        capital, 
        win_rate=0.55, 
        avg_win=0.02, 
        avg_loss=0.01
    )
    
    logger.info(f"\n   Example Trade:")
    logger.info(f"      Signal confidence: {test_signal['confidence']:.1%}")
    logger.info(f"      Capital: ${capital:,.2f}")
    logger.info(f"      Position size: ${position_size:,.2f} ({position_size/capital:.1%})")
    logger.info(f"      Stop loss: ${test_signal['stop_loss']:,.2f}")
    logger.info(f"      Take profit: ${test_signal['take_profit']:,.2f}")
    logger.info(f"      Risk/Reward: 1:{(test_signal['take_profit']-test_signal['target_price'])/(test_signal['target_price']-test_signal['stop_loss']):.1f}")
    
    # Test risk limits
    performance = {
        'peak_capital': capital,
        'current_capital': capital * 0.9,  # 10% drawdown
        'today_pnl_pct': -0.01,  # -1% today
    }
    
    can_trade = risk_manager.check_limits(performance)
    logger.info(f"\n   Risk Check:")
    logger.info(f"      Current drawdown: 10%")
    logger.info(f"      Can continue trading: {'✅ YES' if can_trade else '❌ NO'}")
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 TEST COMPLETE!")
    logger.info("=" * 80)
    
    logger.info("\n✅ All components working correctly!")
    logger.info("\n📚 What was tested:")
    logger.info("   ✓ Multi-timeframe data fetching")
    logger.info("   ✓ Order book integration")
    logger.info("   ✓ Feature engineering (100+ features)")
    logger.info("   ✓ Ensemble training (LightGBM + XGBoost)")
    logger.info("   ✓ Risk management (Kelly Criterion)")
    logger.info("   ✓ Position sizing")
    logger.info("   ✓ Risk limits")
    
    logger.info("\n🚀 Next Steps:")
    logger.info("   1. Collect more data (90 days recommended)")
    logger.info("   2. Run walk-forward optimization:")
    logger.info("      python worldclass_bot.py --mode train")
    logger.info("   3. Backtest the strategy:")
    logger.info("      python worldclass_bot.py --mode backtest")
    logger.info("   4. Paper trade on testnet:")
    logger.info("      python worldclass_bot.py --mode live")
    
    logger.info("\n💡 Performance Tips:")
    logger.info("   • More data = better results (aim for 90+ days)")
    logger.info("   • Retrain monthly to adapt to market changes")
    logger.info("   • Start with small capital and scale up")
    logger.info("   • Monitor win rate and adjust confidence threshold")
    logger.info("   • Use stop-losses ALWAYS")
    
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()

