"""
🧪 TEST & REPORT GENERATOR

Runs paper trading simulation and generates comprehensive report
showing win-rate and profit starting from $10,000.
"""

import sys
import io
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from paper_trade import PaperTradingBot


def generate_report(trades_file: str = 'logs/paper_trades.csv'):
    """Generate comprehensive trading report"""
    
    if not Path(trades_file).exists():
        logger.error(f"Trades file not found: {trades_file}")
        return None
    
    trades_df = pd.read_csv(trades_file)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    # Filter closed trades (those with P&L)
    closed_trades = trades_df[trades_df['pnl'].notna()].copy()
    
    if len(closed_trades) == 0:
        logger.warning("No closed trades found!")
        return None
    
    # Calculate metrics
    starting_balance = 10000.0
    final_balance = trades_df['balance_after'].iloc[-1] if 'balance_after' in trades_df.columns else starting_balance
    total_pnl = final_balance - starting_balance
    total_return_pct = (total_pnl / starting_balance) * 100
    
    # Win rate
    winning_trades = closed_trades[closed_trades['pnl'] > 0]
    losing_trades = closed_trades[closed_trades['pnl'] < 0]
    win_rate = len(winning_trades) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0
    
    # P&L stats
    total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Best/worst trades
    best_trade = closed_trades.loc[closed_trades['pnl'].idxmax()]
    worst_trade = closed_trades.loc[closed_trades['pnl'].idxmin()]
    
    # Generate report
    report = f"""
{'='*80}
📊 TRADING PERFORMANCE REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💰 CAPITAL SUMMARY
{'-'*80}
Starting Balance:        ${starting_balance:,.2f}
Final Balance:           ${final_balance:,.2f}
Total Profit/Loss:       ${total_pnl:+,.2f}
Total Return:            {total_return_pct:+.2f}%
{'='*80}

📈 TRADE STATISTICS
{'-'*80}
Total Trades Executed:   {len(closed_trades)}
Winning Trades:          {len(winning_trades)}
Losing Trades:           {len(losing_trades)}
Win Rate:                {win_rate:.2f}%
{'='*80}

💵 PROFITABILITY ANALYSIS
{'-'*80}
Total Profit:            ${total_profit:,.2f}
Total Loss:              ${total_loss:,.2f}
Net Profit:              ${total_pnl:,.2f}
Profit Factor:           {profit_factor:.2f}
Average Win:             ${avg_win:,.2f}
Average Loss:            ${avg_loss:,.2f}
Risk/Reward Ratio:       {abs(avg_win / avg_loss):.2f}:1 (if avg_loss < 0 else 'N/A')
{'='*80}

🏆 BEST & WORST TRADES
{'-'*80}
BEST TRADE:
  Date:          {best_trade['timestamp']}
  Symbol:        {best_trade['symbol']}
  Entry:         ${best_trade['entry_price']:.2f}
  Exit:          ${best_trade['exit_price']:.2f}
  P&L:           ${best_trade['pnl']:,.2f} ({best_trade['pnl_pct']:.2%})
  Confidence:    {best_trade['confidence']:.4f}

WORST TRADE:
  Date:          {worst_trade['timestamp']}
  Symbol:        {worst_trade['symbol']}
  Entry:         ${worst_trade['entry_price']:.2f}
  Exit:          ${worst_trade['exit_price']:.2f}
  P&L:           ${worst_trade['pnl']:,.2f} ({worst_trade['pnl_pct']:.2%})
  Confidence:    {worst_trade['confidence']:.4f}
{'='*80}

📅 PERFORMANCE TIMELINE
{'-'*80}
"""
    
    # Add timeline of balance changes
    if 'balance_after' in trades_df.columns and 'pnl' in trades_df.columns:
        balance_changes = trades_df[['timestamp', 'balance_after', 'pnl']].copy()
        balance_changes = balance_changes[balance_changes['pnl'].notna()]
        
        if len(balance_changes) > 0:
            report += "Balance Evolution (after each closed trade):\n\n"
            for idx, row in balance_changes.iterrows():
                pnl_str = f"${row['pnl']:+,.2f}" if pd.notna(row['pnl']) else "N/A"
                report += f"  {row['timestamp']}: ${row['balance_after']:,.2f} (P&L: {pnl_str})\n"
    
    report += f"""
{'='*80}
✅ REPORT COMPLETE
{'='*80}

NOTE: This is PAPER TRADING - no real money was risked.
Actual performance may vary due to:
- Slippage in real orders
- Exchange fees (0.1% per trade)
- Network latency
- Market conditions

Starting Capital: ${starting_balance:,.2f}
Final Capital:     ${final_balance:,.2f}
Total Return:      {total_return_pct:+.2f}%
Win Rate:          {win_rate:.2f}%

{'='*80}
"""
    
    return report, {
        'starting_balance': starting_balance,
        'final_balance': final_balance,
        'total_pnl': total_pnl,
        'total_return_pct': total_return_pct,
        'win_rate': win_rate,
        'total_trades': len(closed_trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }


def main():
    """Run test and generate report"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Trading Bot & Generate Report")
    parser.add_argument('--duration', type=int, default=60, help='Trading duration in minutes')
    parser.add_argument('--interval', type=int, default=30, help='Check interval in seconds')
    parser.add_argument('--skip-test', action='store_true', help='Skip testing, only generate report')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("🧪 TESTING TRADING BOT & GENERATING REPORT")
    logger.info("="*80)
    
    if not args.skip_test:
        logger.info("\n📈 Running paper trading simulation...")
        logger.info(f"   Duration: {args.duration} minutes")
        logger.info(f"   Starting Balance: $10,000.00")
        logger.info(f"   Mode: COMPOUNDING (balance grows with profits)\n")
        
        # Run paper trading
        bot = PaperTradingBot()
        bot.run(duration_minutes=args.duration, check_interval_seconds=args.interval)
    
    # Generate report
    logger.info("\n📊 Generating comprehensive report...")
    result = generate_report()
    
    if result:
        report, metrics = result
        
        # Print report
        print(report)
        
        # Save to file
        report_path = Path('logs/trading_report.txt')
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"\n✅ Report saved to: {report_path}")
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"   Starting: ${metrics['starting_balance']:,.2f}")
        logger.info(f"   Final: ${metrics['final_balance']:,.2f}")
        logger.info(f"   P&L: ${metrics['total_pnl']:+,.2f} ({metrics['total_return_pct']:+.2f}%)")
        logger.info(f"   Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"   Total Trades: {metrics['total_trades']}")
    else:
        logger.error("Failed to generate report - no trades found or file missing")


if __name__ == '__main__':
    main()

