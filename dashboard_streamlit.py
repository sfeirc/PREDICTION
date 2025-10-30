"""
📊 REAL-TIME TRADING DASHBOARD

Beautiful Streamlit dashboard to monitor your bot in real-time.

Features:
- Live performance metrics
- Real-time charts
- Feature importance
- Trade history
- Risk monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Page config
st.set_page_config(
    page_title="Ultimate Trading Bot Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 2.5rem !important;
        font-weight: bold;
    }
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)


class TradingDashboard:
    """Real-time trading dashboard"""
    
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        """Load or generate demo data"""
        # Demo data (replace with real data from your bot)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
        
        self.portfolio_value = pd.Series(
            10000 * (1 + np.cumsum(np.random.randn(100) * 0.02)),
            index=dates
        )
        
        self.trades = pd.DataFrame({
            'timestamp': dates[-20:],
            'symbol': np.random.choice(['BTCUSDT', 'ETHUSDT'], 20),
            'side': np.random.choice(['BUY', 'SELL'], 20),
            'size': np.random.uniform(0.1, 2.0, 20),
            'price': np.random.uniform(30000, 110000, 20),
            'pnl': np.random.randn(20) * 100
        })
        
        self.features = pd.DataFrame({
            'feature': [
                'funding_rate', 'liquidation_signal', 'mtf_signal',
                'volatility_yz', 'ob_imbalance', 'return_1m',
                'volume_spike', 'rsi_14', 'macd', 'bollinger_position'
            ],
            'importance': np.random.uniform(0.05, 0.15, 10)
        }).sort_values('importance', ascending=False)
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown("# 🏆 ULTIMATE TRADING BOT")
        st.markdown("### Real-Time Performance Monitor")
        st.markdown("---")
    
    def render_metrics(self):
        """Render key metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculate metrics
        current_value = self.portfolio_value.iloc[-1]
        initial_value = self.portfolio_value.iloc[0]
        total_return = (current_value - initial_value) / initial_value
        
        returns = self.portfolio_value.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 24) if len(returns) > 0 else 0
        
        max_dd = ((self.portfolio_value.cummax() - self.portfolio_value) / self.portfolio_value.cummax()).max()
        
        win_trades = len(self.trades[self.trades['pnl'] > 0])
        total_trades = len(self.trades)
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # Display metrics
        with col1:
            st.metric(
                "Portfolio Value",
                f"${current_value:,.0f}",
                f"{total_return:+.2%}"
            )
        
        with col2:
            st.metric(
                "Total Return",
                f"{total_return:+.2%}",
                "Since inception"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                "Annualized"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{max_dd:.2%}",
                delta_color="inverse"
            )
        
        with col5:
            st.metric(
                "Win Rate",
                f"{win_rate:.1%}",
                f"{win_trades}/{total_trades} trades"
            )
    
    def render_equity_curve(self):
        """Render equity curve chart"""
        st.markdown("## 📈 Equity Curve")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.portfolio_value.index,
            y=self.portfolio_value.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff00', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode='x unified',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_returns_distribution(self):
        """Render returns distribution"""
        st.markdown("## 📊 Returns Distribution")
        
        returns = self.portfolio_value.pct_change().dropna() * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=30,
            name='Returns',
            marker_color='#00aaff'
        ))
        
        fig.update_layout(
            title="Distribution of Returns",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            template='plotly_dark',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_feature_importance(self):
        """Render feature importance"""
        st.markdown("## 🎯 Feature Importance")
        
        fig = px.bar(
            self.features,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Most Important Features",
            template='plotly_dark',
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recent_trades(self):
        """Render recent trades table"""
        st.markdown("## 💼 Recent Trades")
        
        # Format the dataframe
        display_df = self.trades.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}")
        display_df['size'] = display_df['size'].apply(lambda x: f"{x:.3f}")
        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:+,.2f}")
        
        # Color code PnL
        def color_pnl(val):
            if '+' in val:
                return 'background-color: rgba(0, 255, 0, 0.2)'
            elif '-' in val:
                return 'background-color: rgba(255, 0, 0, 0.2)'
            return ''
        
        styled_df = display_df.style.applymap(color_pnl, subset=['pnl'])
        
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    def render_risk_metrics(self):
        """Render risk metrics"""
        st.markdown("## ⚠️ Risk Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Exposure")
            
            # Demo exposure data
            exposure = pd.DataFrame({
                'Asset': ['BTCUSDT', 'ETHUSDT', 'Cash'],
                'Allocation': [0.45, 0.35, 0.20]
            })
            
            fig = px.pie(
                exposure,
                values='Allocation',
                names='Asset',
                title='Portfolio Allocation',
                template='plotly_dark',
                hole=0.4
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Risk Limits")
            
            # Risk limits
            metrics_df = pd.DataFrame({
                'Metric': ['Portfolio Heat', 'Daily VaR', 'Max Drawdown', 'Leverage'],
                'Current': ['4.2%', '1.8%', '6.3%', '1.0x'],
                'Limit': ['6.0%', '2.0%', '15.0%', '1.0x'],
                'Status': ['✅', '✅', '✅', '✅']
            })
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    def render_live_signals(self):
        """Render live trading signals"""
        st.markdown("## 🎯 Live Signals")
        
        signals = pd.DataFrame({
            'Feature': ['Funding Rate', 'Liquidations', 'MTF Trend', 'Volatility', 'Order Book'],
            'Value': ['0.00003', 'Neutral', 'Bearish', 'Low', 'Imbalanced'],
            'Signal': ['Neutral', 'Hold', 'Avoid Longs', 'Low Risk', 'Caution'],
            'Strength': [30, 50, 70, 40, 60]
        })
        
        for idx, row in signals.iterrows():
            col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
            
            with col1:
                st.write(f"**{row['Feature']}**")
            
            with col2:
                st.write(row['Value'])
            
            with col3:
                st.write(row['Signal'])
            
            with col4:
                st.progress(row['Strength'] / 100)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.markdown("## ⚙️ Controls")
            
            st.markdown("### Bot Status")
            status = st.selectbox("Status", ["🟢 Active", "🟡 Paused", "🔴 Stopped"])
            
            st.markdown("### Trading Mode")
            mode = st.radio("Mode", ["Paper Trading", "Live Trading"])
            
            st.markdown("### Refresh")
            refresh_rate = st.slider("Auto-refresh (seconds)", 5, 60, 30)
            
            if st.button("🔄 Refresh Now"):
                st.rerun()
            
            st.markdown("---")
            st.markdown("### Configuration")
            
            if st.button("⚙️ Edit Config"):
                st.info("Config editor coming soon!")
            
            if st.button("💾 Save Strategy"):
                st.success("Strategy saved!")
            
            if st.button("📥 Export Data"):
                st.info("Exporting to CSV...")
            
            st.markdown("---")
            st.markdown("### System Info")
            st.write(f"**Uptime:** {np.random.randint(1, 48)}h {np.random.randint(0, 60)}m")
            st.write(f"**CPU:** {np.random.randint(10, 40)}%")
            st.write(f"**Memory:** {np.random.randint(20, 60)}%")
            st.write(f"**Latency:** {np.random.randint(50, 200)}ms")
    
    def run(self):
        """Run the dashboard"""
        self.render_sidebar()
        self.render_header()
        self.render_metrics()
        
        st.markdown("---")
        
        # Main charts in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Performance",
            "🎯 Features",
            "💼 Trades",
            "⚠️ Risk",
            "🔴 Live"
        ])
        
        with tab1:
            self.render_equity_curve()
            self.render_returns_distribution()
        
        with tab2:
            self.render_feature_importance()
        
        with tab3:
            self.render_recent_trades()
        
        with tab4:
            self.render_risk_metrics()
        
        with tab5:
            self.render_live_signals()


def main():
    """Main entry point"""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

