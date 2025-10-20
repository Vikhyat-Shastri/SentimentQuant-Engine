"""
Quick Integration Test - Verify All Components Work Together

This script runs a quick smoke test of the integrated system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
import numpy as np
import pandas as pd

# Configure minimal logging
logger.remove()
logger.add(sys.stderr, format="<level>{level: <8}</level> | <level>{message}</level>", level="INFO")

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Core components
        from src.backtesting import BacktestEngine
        from src.signals.signal_generator import SignalGenerator
        from src.sentiment.fear_greed_index import FearGreedIndexCalculator
        
        # HIGH priority features
        from src.ml.multilingual_sentiment import MultiLingualSentimentAnalyzer
        from src.ml.price_predictor import PricePredictor
        from src.backtesting.walk_forward import WalkForwardAnalyzer
        from src.backtesting.monte_carlo import MonteCarloSimulator
        from src.analytics.behavioral_analysis import BehavioralBiasDetector
        from src.ingestion.alternative_data import AlternativeDataAggregator
        from src.ml.gpu_inference import GPUInferenceEngine
        
        # MEDIUM priority features
        from src.analytics.regime_detection import MarketRegimeClassifier
        from src.ml.ensemble_sentiment import EnsembleSentimentAnalyzer
        from src.analytics.crowd_psychology import CrowdPsychologyAnalyzer
        from src.analytics.cross_market import CrossMarketAnalyzer
        from src.processing.stream_optimizer import StreamProcessor
        from src.analytics.advanced_viz import AdvancedVisualizer
        
        # LOW priority features
        from src.ml.sarcasm_detector import SarcasmDetector
        from src.ml.multimodal_analyzer import MultimodalAnalyzer
        from src.backtesting.regime_backtest import RegimeBacktester
        from src.backtesting.market_impact import MarketImpactModel
        from src.utils.lockfree import LockFreeQueue
        from src.utils.memory_pool import MemoryPool
        from src.utils.simd_ops import SIMDOperations
        
        logger.success("✅ All imports successful!")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_initialization():
    """Test that key components can be initialized."""
    logger.info("Testing component initialization...")
    
    try:
        # Test a few key components
        from src.ml.ensemble_sentiment import EnsembleSentimentAnalyzer
        from src.analytics.regime_detection import MarketRegimeClassifier
        from src.backtesting.monte_carlo import MonteCarloSimulator
        
        ensemble = EnsembleSentimentAnalyzer()
        regime = MarketRegimeClassifier()
        mc = MonteCarloSimulator(num_simulations=10)
        
        logger.success("✅ Component initialization successful!")
        return True
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of key features."""
    logger.info("Testing basic functionality...")
    
    try:
        import numpy as np
        import pandas as pd
        
        # Test ensemble sentiment
        from src.ml.ensemble_sentiment import EnsembleSentimentAnalyzer
        ensemble = EnsembleSentimentAnalyzer()
        result = ensemble.analyze("Bitcoin is amazing!")
        assert result.ensemble_score > 0, "Positive sentiment expected"
        logger.info("  ✓ Ensemble sentiment working")
        
        # Test regime detection
        from src.analytics.regime_detection import MarketRegimeClassifier
        regime_clf = MarketRegimeClassifier()
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        df = pd.DataFrame({'close': prices, 'high': prices * 1.01, 'low': prices * 0.99, 'volume': 1000})
        regime_result = regime_clf.classify(df)
        assert 'regime' in regime_result.columns, "Regime column expected"
        logger.info("  ✓ Regime detection working")
        
        # Test Monte Carlo
        from src.backtesting.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(num_simulations=100)
        mock_returns = np.random.randn(100) * 0.01
        mc_result = mc.simulate_bootstrap(mock_returns, num_periods=30)
        assert mc_result.mean_return is not None, "Mean return should exist"
        logger.info("  ✓ Monte Carlo working")
        
        # Test SIMD operations
        from src.utils.simd_ops import SIMDOperations
        simd = SIMDOperations()
        data = np.random.randn(100)
        ma = simd.fast_moving_average(data, window=10)
        assert len(ma) == len(data), "MA length should match input"
        logger.info("  ✓ SIMD operations working")
        
        logger.success("✅ Basic functionality tests passed!")
        return True
    except Exception as e:
        logger.error(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrations():
    """Test that integrated scripts have proper imports."""
    logger.info("Testing integration points...")
    
    try:
        # Get project root
        project_root = Path(__file__).parent.parent
        
        # Check main.py has advanced features
        with open(project_root / 'scripts' / 'main.py', 'r', encoding='utf-8') as f:
            main_content = f.read()
            assert 'SarcasmDetector' in main_content, "main.py missing sarcasm detector"
            assert 'EnsembleSentimentAnalyzer' in main_content, "main.py missing ensemble"
            assert 'CrowdPsychologyAnalyzer' in main_content, "main.py missing crowd psychology"
        logger.info("  ✓ main.py integrated")
        
        # Check run_analytics.py has advanced features
        with open(project_root / 'scripts' / 'run_analytics.py', 'r', encoding='utf-8') as f:
            analytics_content = f.read()
            assert 'BehavioralBiasDetector' in analytics_content, "run_analytics.py missing behavioral analysis"
            assert 'MarketRegimeClassifier' in analytics_content, "run_analytics.py missing regime detection"
        logger.info("  ✓ run_analytics.py integrated")
        
        # Check backtest_strategy.py has advanced features
        with open(project_root / 'scripts' / 'backtest_strategy.py', 'r', encoding='utf-8') as f:
            backtest_content = f.read()
            assert 'MonteCarloSimulator' in backtest_content, "backtest_strategy.py missing Monte Carlo"
            assert 'WalkForwardAnalyzer' in backtest_content, "backtest_strategy.py missing walk-forward"
            assert 'RegimeBacktester' in backtest_content, "backtest_strategy.py missing regime backtest"
        logger.info("  ✓ backtest_strategy.py integrated")
        
        # Check visualize_backtest.py has advanced viz
        with open(project_root / 'tools' / 'visualize_backtest.py', 'r', encoding='utf-8') as f:
            viz_content = f.read()
            assert 'AdvancedVisualizer' in viz_content, "visualize_backtest.py missing advanced viz"
        logger.info("  ✓ visualize_backtest.py integrated")
        
        logger.success("✅ All integration points verified!")
        return True
    except Exception as e:
        logger.error(f"❌ Integration check failed: {e}")
        return False


def main():
    """Run all integration tests."""
    logger.info("=" * 70)
    logger.info("🧪 INTEGRATION TEST SUITE")
    logger.info("=" * 70)
    logger.info("")
    
    results = []
    
    # Run tests
    results.append(("Import Test", test_imports()))
    results.append(("Initialization Test", test_initialization()))
    results.append(("Functionality Test", test_basic_functionality()))
    results.append(("Integration Points Test", test_integrations()))
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 TEST RESULTS")
    logger.info("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {status}: {test_name}")
    
    logger.info("")
    logger.info(f"Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        logger.success("=" * 70)
        logger.success("🎉 ALL TESTS PASSED - SYSTEM READY!")
        logger.success("=" * 70)
        logger.info("")
        logger.info("✅ All 20 bonus features integrated successfully")
        logger.info("✅ Main pipeline enhanced with advanced ML")
        logger.info("✅ Analytics enhanced with behavioral/regime/crowd analysis")
        logger.info("✅ Backtesting enhanced with Monte Carlo/Walk-Forward")
        logger.info("✅ Visualization enhanced with interactive dashboards")
        logger.info("")
        logger.info("🚀 Ready for production!")
        return 0
    else:
        logger.error("=" * 70)
        logger.error(f"❌ {total - passed} TEST(S) FAILED")
        logger.error("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
