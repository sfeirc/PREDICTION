"""
Test the confidence boost effect
"""

import numpy as np

def boost_prediction(proba_raw, boost_factor=2.5):
    """Apply confidence boost to prediction"""
    if proba_raw > 0.5:
        normalized = (proba_raw - 0.5) / 0.5
        boosted = normalized ** (1 / boost_factor)
        proba = 0.5 + boosted * 0.5
    else:
        normalized = (0.5 - proba_raw) / 0.5
        boosted = normalized ** (1 / boost_factor)
        proba = 0.5 - boosted * 0.5
    return max(0.0, min(1.0, proba))

print("="*80)
print("🎯 CONFIDENCE BOOST TEST")
print("="*80)
print("\nOriginal → Boosted (Boost Factor: 2.5)")
print("-"*80)

test_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for orig in test_values:
    boosted = boost_prediction(orig)
    change = boosted - orig
    direction = "🟢 BUY" if boosted > 0.5 else "🔴 SELL"
    confidence = abs(boosted - 0.5) * 2
    
    print(f"  {orig:.2f} → {boosted:.4f} ({change:+.4f}) | {direction} | Confidence: {confidence:.2%}")

print("\n" + "="*80)
print("✅ Predictions are now pushed toward extremes (closer to 0 or 1)")
print("   This makes BUY/SELL signals more confident!")
print("="*80)

