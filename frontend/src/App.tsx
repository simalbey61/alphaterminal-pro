/**
 * AlphaTerminal Pro - Main Application
 * =====================================
 * 
 * Root application component.
 */

import React, { useState, useEffect } from 'react';
import './styles/globals.css';
import {
  DashboardLayout,
  DashboardHeader,
  Sidebar,
  PortfolioOverview,
  ChartPanel,
  Watchlist,
} from './components/dashboard';
import { SignalsList, SignalsSummary } from './components/signals';
import { BacktestForm, BacktestResults, BacktestHistory } from './components/backtest';
import { Card } from './components/common';
import { useTheme, useLocalStorage } from './hooks';
import type { 
  TimeFrame, 
  TradingSignal, 
  OHLCVBar, 
  BacktestResult,
  PortfolioSummary,
  SignalType,
  PositionSide,
  BacktestStatus
} from './types';

// =============================================================================
// MOCK DATA GENERATORS
// =============================================================================

const generateMockOHLCV = (days: number): OHLCVBar[] => {
  const data: OHLCVBar[] = [];
  let price = 100;
  
  for (let i = 0; i < days; i++) {
    const change = (Math.random() - 0.5) * 4;
    const open = price;
    const close = price + change;
    const high = Math.max(open, close) + Math.random() * 2;
    const low = Math.min(open, close) - Math.random() * 2;
    
    data.push({
      timestamp: new Date(Date.now() - (days - i) * 86400000).toISOString(),
      open,
      high,
      low,
      close,
      volume: Math.floor(Math.random() * 1000000) + 100000,
    });
    
    price = close;
  }
  
  return data;
};

const mockSignals: TradingSignal[] = [
  {
    id: '1',
    symbol: 'THYAO',
    signalType: SignalType.BUY,
    price: 285.50,
    targetPrice: 310.00,
    stopLoss: 270.00,
    confidence: 0.85,
    strategy: 'SMA Crossover',
    timeframe: TimeFrame.D1,
    createdAt: new Date().toISOString(),
    indicators: { RSI: 42, MACD: 0.15, ADX: 28 },
  },
  {
    id: '2',
    symbol: 'GARAN',
    signalType: SignalType.SELL,
    price: 48.20,
    targetPrice: 44.00,
    stopLoss: 50.50,
    confidence: 0.72,
    strategy: 'RSI Divergence',
    timeframe: TimeFrame.H4,
    createdAt: new Date(Date.now() - 3600000).toISOString(),
    indicators: { RSI: 75, MACD: -0.08 },
  },
  {
    id: '3',
    symbol: 'AKBNK',
    signalType: SignalType.BUY,
    price: 15.80,
    targetPrice: 17.50,
    stopLoss: 15.00,
    confidence: 0.68,
    strategy: 'Breakout',
    timeframe: TimeFrame.D1,
    createdAt: new Date(Date.now() - 7200000).toISOString(),
  },
];

const mockPortfolio: PortfolioSummary = {
  totalValue: 125000,
  cashBalance: 45000,
  positionsValue: 80000,
  unrealizedPnl: 5200,
  realizedPnl: 12000,
  dailyPnl: 1250,
  dailyPnlPercent: 1.01,
  positions: [
    {
      symbol: 'THYAO',
      side: PositionSide.LONG,
      quantity: 100,
      entryPrice: 275.00,
      currentPrice: 285.50,
      unrealizedPnl: 1050,
      unrealizedPnlPercent: 3.82,
      marketValue: 28550,
      openedAt: new Date(Date.now() - 5 * 86400000).toISOString(),
    },
    {
      symbol: 'GARAN',
      side: PositionSide.LONG,
      quantity: 500,
      entryPrice: 46.50,
      currentPrice: 48.20,
      unrealizedPnl: 850,
      unrealizedPnlPercent: 3.66,
      marketValue: 24100,
      openedAt: new Date(Date.now() - 3 * 86400000).toISOString(),
    },
  ],
};

const mockWatchlist = [
  { symbol: 'THYAO', price: 285.50, change: 3.50, changePercent: 1.24, sparkline: generateMockOHLCV(20).map(d => d.close) },
  { symbol: 'GARAN', price: 48.20, change: -0.80, changePercent: -1.63, sparkline: generateMockOHLCV(20).map(d => d.close) },
  { symbol: 'AKBNK', price: 15.80, change: 0.25, changePercent: 1.61, sparkline: generateMockOHLCV(20).map(d => d.close) },
  { symbol: 'EREGL', price: 52.40, change: 1.10, changePercent: 2.14, sparkline: generateMockOHLCV(20).map(d => d.close) },
  { symbol: 'ASELS', price: 78.90, change: -1.20, changePercent: -1.50, sparkline: generateMockOHLCV(20).map(d => d.close) },
];

// =============================================================================
// MAIN APP
// =============================================================================

const App: React.FC = () => {
  const { theme, toggleTheme } = useTheme();
  const [sidebarCollapsed, setSidebarCollapsed] = useLocalStorage('sidebar-collapsed', false);
  const [activeSection, setActiveSection] = useState('dashboard');
  
  const [selectedSymbol, setSelectedSymbol] = useState('THYAO');
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeFrame>(TimeFrame.D1);
  
  const [chartData, setChartData] = useState<OHLCVBar[]>([]);
  const [signals] = useState<TradingSignal[]>(mockSignals);
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  
  // Load chart data
  useEffect(() => {
    setChartData(generateMockOHLCV(100));
  }, [selectedSymbol, selectedTimeframe]);

  // Handle backtest
  const handleRunBacktest = async (config: unknown) => {
    setBacktestLoading(true);
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const mockResult: BacktestResult = {
      id: Math.random().toString(36).substr(2, 9),
      status: BacktestStatus.COMPLETED,
      config: config as BacktestResult['config'],
      totalReturn: 25000,
      totalReturnPct: 25.0,
      sharpeRatio: 1.52,
      sortinoRatio: 2.10,
      calmarRatio: 2.08,
      maxDrawdown: 0.12,
      volatility: 0.18,
      totalTrades: 45,
      winningTrades: 27,
      losingTrades: 18,
      winRate: 0.60,
      profitFactor: 1.85,
      avgTrade: 555,
      avgWinner: 1200,
      avgLoser: -600,
      largestWinner: 5000,
      largestLoser: -2000,
      avgHoldingHours: 48,
      equityCurve: Array.from({ length: 252 }, (_, i) => 100000 + Math.random() * 500 * i + (Math.random() - 0.3) * 5000),
      trades: Array.from({ length: 45 }, (_, i) => ({
        id: `trade-${i}`,
        entryTime: new Date(Date.now() - (45 - i) * 86400000 * 5).toISOString(),
        exitTime: new Date(Date.now() - (45 - i) * 86400000 * 5 + 86400000 * 2).toISOString(),
        side: Math.random() > 0.5 ? PositionSide.LONG : PositionSide.SHORT,
        entryPrice: 100 + Math.random() * 20,
        exitPrice: 100 + Math.random() * 25,
        quantity: 100,
        pnl: (Math.random() - 0.4) * 2000,
        pnlPercent: (Math.random() - 0.4) * 10,
        holdingHours: Math.floor(Math.random() * 72) + 12,
        exitReason: Math.random() > 0.5 ? 'Take Profit' : 'Stop Loss',
      })),
      createdAt: new Date().toISOString(),
      completedAt: new Date().toISOString(),
    };
    
    setBacktestResult(mockResult);
    setBacktestLoading(false);
  };

  return (
    <DashboardLayout
      sidebar={
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
          activeSection={activeSection}
          onSectionChange={setActiveSection}
        />
      }
      header={
        <DashboardHeader
          title="AlphaTerminal Pro"
          symbol={selectedSymbol}
          onSymbolChange={setSelectedSymbol}
          selectedTimeframe={selectedTimeframe}
          onTimeframeChange={setSelectedTimeframe}
        />
      }
    >
      {activeSection === 'dashboard' && (
        <div className="dashboard-view">
          <PortfolioOverview portfolio={mockPortfolio} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '16px', marginTop: '16px' }}>
            <ChartPanel
              data={chartData}
              signals={signals.filter(s => s.symbol === selectedSymbol)}
              symbol={selectedSymbol}
              timeframe={selectedTimeframe}
            />
            <div>
              <Watchlist items={mockWatchlist} selectedSymbol={selectedSymbol} onItemClick={setSelectedSymbol} />
              <div style={{ marginTop: '16px' }}>
                <SignalsSummary signals={signals} />
              </div>
            </div>
          </div>
        </div>
      )}

      {activeSection === 'backtest' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <BacktestForm onSubmit={handleRunBacktest} loading={backtestLoading} />
            <BacktestHistory backtests={backtestResult ? [backtestResult] : []} onSelect={setBacktestResult} />
          </div>
          {backtestResult && <div style={{ marginTop: '16px' }}><BacktestResults result={backtestResult} /></div>}
        </div>
      )}

      {activeSection === 'signals' && (
        <div>
          <SignalsSummary signals={signals} />
          <Card title="Active Signals"><SignalsList signals={signals} /></Card>
        </div>
      )}

      {activeSection === 'portfolio' && (
        <div>
          <PortfolioOverview portfolio={mockPortfolio} />
          <Card title="Open Positions" style={{ marginTop: '16px' }}>
            <table className="table">
              <thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Current</th><th>P&L</th></tr></thead>
              <tbody>
                {mockPortfolio.positions.map(pos => (
                  <tr key={pos.symbol}>
                    <td><strong>{pos.symbol}</strong></td>
                    <td>{pos.side.toUpperCase()}</td>
                    <td>{pos.quantity}</td>
                    <td>${pos.entryPrice.toFixed(2)}</td>
                    <td>${pos.currentPrice.toFixed(2)}</td>
                    <td className={pos.unrealizedPnl >= 0 ? 'positive' : 'negative'}>
                      ${pos.unrealizedPnl.toFixed(2)} ({pos.unrealizedPnlPercent.toFixed(2)}%)
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {activeSection === 'settings' && (
        <Card title="⚙️ Settings">
          <div style={{ padding: '20px' }}>
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>Theme</label>
              <button onClick={toggleTheme} className="btn btn-secondary">
                {theme === 'dark' ? '🌙 Dark Mode' : '☀️ Light Mode'}
              </button>
            </div>
          </div>
        </Card>
      )}
    </DashboardLayout>
  );
};

export default App;
