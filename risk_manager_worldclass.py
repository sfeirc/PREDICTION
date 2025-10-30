"""
Professional Risk Management System
Inspired by: Proprietary trading firms, hedge funds
"""

import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class RiskManagerWorldClass:
    """
    Enterprise-grade risk management with Kelly Criterion, portfolio heat, and drawdown protection.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_config = config['risk']
        
        # Track performance for dynamic risk adjustment
        self.trade_history = []
        self.daily_pnl = []
        
        logger.info("🛡️  Risk Manager initialized")
        logger.info(f"   Max Position Size: {self.risk_config['max_position_size']:.1%}")
        logger.info(f"   Stop Loss: {self.risk_config['stop_loss']:.1%}")
        logger.info(f"   Take Profit: {self.risk_config['take_profit']:.1%}")
        logger.info(f"   Max Drawdown: {self.risk_config['max_drawdown']:.1%}")
    
    def calculate_position_size(self, signal: Dict, capital: float, win_rate: float = None, avg_win: float = None, avg_loss: float = None) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            signal: Trading signal with confidence
            capital: Available capital
            win_rate: Historical win rate (optional)
            avg_win: Average win size (optional)
            avg_loss: Average loss size (optional)
        
        Returns:
            Position size in USD
        """
        sizing_method = self.risk_config['sizing_method']
        
        if sizing_method == 'kelly' and win_rate and avg_win and avg_loss:
            # Kelly Criterion: f* = (p*b - q) / b
            # where p = win rate, q = 1-p, b = avg_win/avg_loss
            b = abs(avg_win / avg_loss) if avg_loss != 0 else 2.0
            q = 1 - win_rate
            kelly = (win_rate * b - q) / b
            
            # Use fractional Kelly for safety
            kelly_fraction = self.risk_config['kelly_fraction']
            kelly = max(0, min(kelly * kelly_fraction, self.risk_config['max_position_size']))
            
            position_size = capital * kelly
            
            logger.debug(f"Kelly sizing: {kelly:.1%} of capital")
        
        else:
            # Fixed fractional sizing
            position_size = capital * 0.1  # Default 10%
        
        # Adjust based on confidence
        confidence_mult = self._confidence_multiplier(signal['confidence'])
        position_size *= confidence_mult
        
        # Cap at max position size
        max_pos = capital * self.risk_config['max_position_size']
        position_size = min(position_size, max_pos)
        
        return position_size
    
    def _confidence_multiplier(self, confidence: float) -> float:
        """
        Adjust position size based on model confidence.
        
        Confidence ranges:
        - 0.50-0.60: 0.3x
        - 0.60-0.70: 0.5x
        - 0.70-0.80: 0.8x
        - 0.80-0.90: 1.0x
        - 0.90+:     1.2x
        """
        if confidence < 0.60:
            return 0.3
        elif confidence < 0.70:
            return 0.5
        elif confidence < 0.80:
            return 0.8
        elif confidence < 0.90:
            return 1.0
        else:
            return 1.2
    
    def check_limits(self, performance: Dict) -> bool:
        """
        Check if risk limits are breached.
        
        Returns:
            True if safe to continue trading
        """
        # Check max drawdown
        peak_capital = performance.get('peak_capital', self.risk_config['initial_capital'])
        current_capital = performance.get('current_capital', self.risk_config['initial_capital'])
        drawdown = (peak_capital - current_capital) / peak_capital
        
        if drawdown >= self.risk_config['max_drawdown']:
            logger.warning(f"⚠️  Max drawdown breached: {drawdown:.1%}")
            return False
        
        # Check daily loss limit
        today_pnl_pct = performance.get('today_pnl_pct', 0)
        if today_pnl_pct <= -self.risk_config['max_daily_loss']:
            logger.warning(f"⚠️  Daily loss limit breached: {today_pnl_pct:.1%}")
            return False
        
        return True
    
    def approve_trade(self, signal: Dict, positions: Dict) -> bool:
        """
        Approve or reject a trade signal based on risk rules.
        
        Args:
            signal: Trading signal
            positions: Current open positions
        
        Returns:
            True if trade is approved
        """
        # Check confidence threshold
        if signal['confidence'] < self.risk_config['min_confidence']:
            logger.debug(f"❌ Trade rejected: confidence {signal['confidence']:.1%} < {self.risk_config['min_confidence']:.1%}")
            return False
        
        # Check portfolio heat (total risk exposure)
        total_risk = sum(pos.get('risk_amount', 0) for pos in positions.values())
        max_heat = self.risk_config['initial_capital'] * self.risk_config['max_portfolio_heat']
        
        if total_risk >= max_heat:
            logger.debug(f"❌ Trade rejected: portfolio heat {total_risk:.0f} >= {max_heat:.0f}")
            return False
        
        # Check correlation exposure (max 30% in correlated assets)
        if len(positions) > 0:
            correlated_exposure = self._calculate_correlated_exposure(signal['symbol'], positions)
            if correlated_exposure > self.risk_config['max_correlation_exposure']:
                logger.debug(f"❌ Trade rejected: correlated exposure {correlated_exposure:.1%}")
                return False
        
        return True
    
    def _calculate_correlated_exposure(self, symbol: str, positions: Dict) -> float:
        """Calculate exposure to correlated assets."""
        # Simplified: assume BTC/ETH/BNB are correlated
        correlated_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        if symbol not in correlated_symbols:
            return 0.0
        
        total_exposure = sum(
            pos['size'] for pos in positions.values()
            if pos['symbol'] in correlated_symbols
        )
        
        return total_exposure / self.risk_config['initial_capital']
    
    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price."""
        stop_pct = self.risk_config['stop_loss']
        
        if side == 'BUY':
            return entry_price * (1 - stop_pct)
        else:
            return entry_price * (1 + stop_pct)
    
    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price."""
        tp_pct = self.risk_config['take_profit']
        
        if side == 'BUY':
            return entry_price * (1 + tp_pct)
        else:
            return entry_price * (1 - tp_pct)
    
    def record_trade(self, trade: Dict):
        """Record trade for performance tracking."""
        self.trade_history.append(trade)
    
    def get_statistics(self) -> Dict:
        """Calculate trading statistics."""
        if not self.trade_history:
            return {}
        
        trades = [t for t in self.trade_history if t.get('closed', False)]
        
        if not trades:
            return {}
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0
        
        total_pnl = sum(t['pnl'] for t in trades)
        
        # Profit factor
        total_wins = sum(t['pnl'] for t in wins)
        total_losses = abs(sum(t['pnl'] for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
        }

