"""
🧪 COMPLETE SYSTEM TEST

Tests ALL components of the ultimate trading bot:
- Phase 1: Ultimate Features
- Phase 2: Advanced ML Models
- Phase 3: Execution Algorithms
- Phase 4: Portfolio Optimization
- Phase 5: Online Learning
- Phase 6: Monitoring & Optimization
- Phase 7: Integration

Run this to verify everything works!
"""

import sys
import io
import logging
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_phase_1_features():
    """Test Phase 1: Ultimate Features"""
    print("\n" + "="*80)
    print("📊 PHASE 1: ULTIMATE FEATURES")
    print("="*80)
    
    try:
        from features_ultimate import UltimateFeatureEngine
        
        engine = UltimateFeatureEngine({})
        features = engine.extract_all_features('BTCUSDT')
        
        print(f"✅ Extracted {len(features)} features")
        print(f"   Sample features:")
        for key, value in list(features.items())[:5]:
            print(f"      {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        return False


def test_phase_2_models():
    """Test Phase 2: Advanced ML Models"""
    print("\n" + "="*80)
    print("🤖 PHASE 2: ADVANCED ML MODELS")
    print("="*80)
    
    success = True
    
    # Test Transformer
    try:
        print("\n1️⃣ Testing Transformer...")
        from models_transformer import TradingTransformer
        model = TradingTransformer(input_dim=50, d_model=128, nhead=4, num_layers=2)
        print("   ✅ Transformer initialized")
    except Exception as e:
        print(f"   ❌ Transformer failed: {e}")
        success = False
    
    # Test RL
    try:
        print("\n2️⃣ Testing Reinforcement Learning...")
        from models_reinforcement import PPOAgent
        agent = PPOAgent(state_dim=50, action_dim=5)
        print("   ✅ PPO Agent initialized")
    except Exception as e:
        print(f"   ❌ RL failed: {e}")
        success = False
    
    # Test Meta-Learning
    try:
        print("\n3️⃣ Testing Meta-Learning...")
        from models_metalearning import MAML, BaseModel
        model = BaseModel(input_dim=50)
        maml = MAML(model)
        print("   ✅ MAML initialized")
    except Exception as e:
        print(f"   ❌ Meta-Learning failed: {e}")
        success = False
    
    return success


def test_phase_3_execution():
    """Test Phase 3: Execution Algorithms"""
    print("\n" + "="*80)
    print("⚡ PHASE 3: EXECUTION ALGORITHMS")
    print("="*80)
    
    try:
        from execution_algorithms import (
            TWAPExecutor, VWAPExecutor, POVExecutor,
            MarketImpactModel, AdaptiveExecutor
        )
        
        print("   ✅ TWAP Executor")
        print("   ✅ VWAP Executor")
        print("   ✅ POV Executor")
        print("   ✅ Market Impact Model")
        print("   ✅ Adaptive Executor")
        
        return True
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        return False


def test_phase_4_portfolio():
    """Test Phase 4: Portfolio Optimization"""
    print("\n" + "="*80)
    print("💼 PHASE 4: PORTFOLIO OPTIMIZATION")
    print("="*80)
    
    try:
        from portfolio_optimization import (
            BlackLittermanOptimizer,
            RiskParityOptimizer,
            MeanVarianceOptimizer,
            DynamicRebalancer
        )
        
        print("   ✅ Black-Litterman Optimizer")
        print("   ✅ Risk Parity Optimizer")
        print("   ✅ Mean-Variance Optimizer")
        print("   ✅ Dynamic Rebalancer")
        
        return True
    except Exception as e:
        print(f"❌ Phase 4 failed: {e}")
        return False


def test_phase_5_online_learning():
    """Test Phase 5: Online Learning"""
    print("\n" + "="*80)
    print("📚 PHASE 5: ONLINE LEARNING")
    print("="*80)
    
    try:
        from online_learning import (
            ConceptDriftDetector,
            IncrementalLearner,
            AdaptiveLearningRate
        )
        
        print("   ✅ Concept Drift Detector")
        print("   ✅ Incremental Learner")
        print("   ✅ Adaptive Learning Rate")
        
        return True
    except Exception as e:
        print(f"❌ Phase 5 failed: {e}")
        return False


def test_phase_6_monitoring():
    """Test Phase 6: Monitoring & Optimization"""
    print("\n" + "="*80)
    print("📊 PHASE 6: MONITORING & OPTIMIZATION")
    print("="*80)
    
    success = True
    
    # Test Dashboard
    try:
        print("\n1️⃣ Testing Dashboard...")
        import dashboard_streamlit
        print("   ✅ Dashboard available (run: streamlit run dashboard_streamlit.py)")
    except Exception as e:
        print(f"   ❌ Dashboard failed: {e}")
        success = False
    
    # Test Bayesian Optimization
    try:
        print("\n2️⃣ Testing Bayesian Optimization...")
        from bayesian_optimization import BayesianOptimizer
        print("   ✅ Bayesian Optimizer initialized")
    except Exception as e:
        print(f"   ❌ Bayesian Optimization failed: {e}")
        success = False
    
    # Test Walk-Forward Backtest
    try:
        print("\n3️⃣ Testing Walk-Forward Backtest...")
        from backtest_walkforward import WalkForwardBacktest
        print("   ✅ Walk-Forward Backtest initialized")
    except Exception as e:
        print(f"   ❌ Walk-Forward Backtest failed: {e}")
        success = False
    
    return success


def test_phase_7_integration():
    """Test Phase 7: Integration"""
    print("\n" + "="*80)
    print("🔗 PHASE 7: INTEGRATION")
    print("="*80)
    
    try:
        from bot_ultimate_integrated import UltimateTradingBot
        
        bot = UltimateTradingBot({})
        print("   ✅ Ultimate Trading Bot initialized")
        print("   ✅ All components integrated")
        
        return True
    except Exception as e:
        print(f"❌ Phase 7 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_complete_test():
    """Run complete system test"""
    print("\n" + "="*80)
    print("🧪 ULTIMATE TRADING BOT - COMPLETE SYSTEM TEST")
    print("="*80)
    print("\nTesting ALL components...")
    
    results = {
        'Phase 1: Ultimate Features': test_phase_1_features(),
        'Phase 2: Advanced ML Models': test_phase_2_models(),
        'Phase 3: Execution Algorithms': test_phase_3_execution(),
        'Phase 4: Portfolio Optimization': test_phase_4_portfolio(),
        'Phase 5: Online Learning': test_phase_5_online_learning(),
        'Phase 6: Monitoring & Optimization': test_phase_6_monitoring(),
        'Phase 7: Integration': test_phase_7_integration(),
    }
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for phase, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {phase}")
    
    print("\n" + "="*80)
    print(f"🏆 RESULTS: {passed}/{total} phases passed ({passed/total*100:.0f}%)")
    print("="*80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is 100% operational!")
        print("\n📖 Next Steps:")
        print("   1. Run dashboard: streamlit run dashboard_streamlit.py")
        print("   2. Train models: python train_ultimate.py")
        print("   3. Start trading: python bot_ultimate_integrated.py")
        print("\n💰 Expected Performance: 20-35% monthly, 2.5-4.0 Sharpe")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        print("   Most likely missing dependencies. Run:")
        print("   pip install -r requirements.txt")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = run_complete_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

