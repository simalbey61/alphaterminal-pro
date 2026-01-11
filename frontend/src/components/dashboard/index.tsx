/**
 * AlphaTerminal Pro - Dashboard Components
 * =========================================
 * 
 * Main dashboard layout and widgets.
 */

import React, { useState } from 'react';
import { Card, MetricCard, Tabs, Badge, LoadingState, ErrorState } from '../common';
import { CandlestickChart, EquityCurveChart, MiniChart } from '../charts';
import { SignalsList } from '../signals';
import type { 
  PortfolioSummary, 
  TradingSignal, 
  OHLCVBar, 
  TimeFrame, 
  Position 
} from '../../types';

// =============================================================================
// DASHBOARD HEADER
// =============================================================================

interface DashboardHeaderProps {
  title: string;
  symbol?: string;
  onSymbolChange?: (symbol: string) => void;
  onTimeframeChange?: (tf: TimeFrame) => void;
  selectedTimeframe: TimeFrame;
}

export const DashboardHeader: React.FC<DashboardHeaderProps> = ({
  title,
  symbol,
  onSymbolChange,
  onTimeframeChange,
  selectedTimeframe,
}) => {
  const [searchValue, setSearchValue] = useState(symbol || '');

  const timeframes: { value: TimeFrame; label: string }[] = [
    { value: '1m' as TimeFrame, label: '1m' },
    { value: '5m' as TimeFrame, label: '5m' },
    { value: '15m' as TimeFrame, label: '15m' },
    { value: '1h' as TimeFrame, label: '1H' },
    { value: '4h' as TimeFrame, label: '4H' },
    { value: '1d' as TimeFrame, label: '1D' },
    { value: '1w' as TimeFrame, label: '1W' },
  ];

  return (
    <header className="dashboard-header">
      <div className="header-left">
        <h1 className="header-title">{title}</h1>
        {symbol && <span className="header-symbol">{symbol}</span>}
      </div>

      <div className="header-center">
        <div className="symbol-search">
          <input
            type="text"
            placeholder="Search symbol..."
            value={searchValue}
            onChange={(e) => setSearchValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && searchValue) {
                onSymbolChange?.(searchValue.toUpperCase());
              }
            }}
            className="search-input"
          />
        </div>

        <div className="timeframe-selector">
          {timeframes.map((tf) => (
            <button
              key={tf.value}
              className={`tf-btn ${selectedTimeframe === tf.value ? 'active' : ''}`}
              onClick={() => onTimeframeChange?.(tf.value)}
            >
              {tf.label}
            </button>
          ))}
        </div>
      </div>

      <div className="header-right">
        <button className="icon-btn" title="Notifications">
          🔔
        </button>
        <button className="icon-btn" title="Settings">
          ⚙️
        </button>
      </div>
    </header>
  );
};

// =============================================================================
// PORTFOLIO OVERVIEW
// =============================================================================

interface PortfolioOverviewProps {
  portfolio: PortfolioSummary | null;
  loading?: boolean;
  error?: string;
}

export const PortfolioOverview: React.FC<PortfolioOverviewProps> = ({
  portfolio,
  loading,
  error,
}) => {
  if (loading) return <LoadingState message="Loading portfolio..." />;
  if (error) return <ErrorState message={error} />;
  if (!portfolio) return null;

  return (
    <div className="portfolio-overview">
      <div className="metrics-row">
        <MetricCard
          label="Portfolio Value"
          value={portfolio.totalValue}
          prefix="$"
          change={portfolio.dailyPnlPercent}
          icon="💰"
        />
        <MetricCard
          label="Daily P&L"
          value={portfolio.dailyPnl}
          prefix="$"
          change={portfolio.dailyPnlPercent}
          icon="📈"
        />
        <MetricCard
          label="Unrealized P&L"
          value={portfolio.unrealizedPnl}
          prefix="$"
          icon="📊"
        />
        <MetricCard
          label="Cash Balance"
          value={portfolio.cashBalance}
          prefix="$"
          icon="💵"
        />
      </div>
    </div>
  );
};

// =============================================================================
// POSITIONS TABLE
// =============================================================================

interface PositionsTableProps {
  positions: Position[];
  onPositionClick?: (position: Position) => void;
}

export const PositionsTable: React.FC<PositionsTableProps> = ({
  positions,
  onPositionClick,
}) => {
  if (positions.length === 0) {
    return (
      <div className="empty-positions">
        <p>No open positions</p>
      </div>
    );
  }

  return (
    <div className="positions-table">
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Side</th>
            <th>Qty</th>
            <th>Entry</th>
            <th>Current</th>
            <th>P&L</th>
            <th>P&L %</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((pos) => (
            <tr
              key={pos.symbol}
              onClick={() => onPositionClick?.(pos)}
              className="position-row"
            >
              <td className="symbol-cell">
                <strong>{pos.symbol}</strong>
              </td>
              <td>
                <Badge variant={pos.side === 'long' ? 'success' : 'danger'}>
                  {pos.side.toUpperCase()}
                </Badge>
              </td>
              <td>{pos.quantity}</td>
              <td>${pos.entryPrice.toFixed(2)}</td>
              <td>${pos.currentPrice.toFixed(2)}</td>
              <td className={pos.unrealizedPnl >= 0 ? 'positive' : 'negative'}>
                ${pos.unrealizedPnl.toFixed(2)}
              </td>
              <td className={pos.unrealizedPnlPercent >= 0 ? 'positive' : 'negative'}>
                {pos.unrealizedPnlPercent >= 0 ? '+' : ''}
                {pos.unrealizedPnlPercent.toFixed(2)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// =============================================================================
// CHART PANEL
// =============================================================================

interface ChartPanelProps {
  data: OHLCVBar[];
  signals?: TradingSignal[];
  symbol: string;
  timeframe: TimeFrame;
  loading?: boolean;
  error?: string;
}

export const ChartPanel: React.FC<ChartPanelProps> = ({
  data,
  signals,
  symbol,
  timeframe,
  loading,
  error,
}) => {
  const [activeTab, setActiveTab] = useState('chart');

  const tabs = [
    { id: 'chart', label: 'Chart' },
    { id: 'signals', label: 'Signals' },
    { id: 'info', label: 'Info' },
  ];

  if (loading) return <LoadingState message="Loading chart data..." />;
  if (error) return <ErrorState message={error} />;

  return (
    <Card className="chart-panel">
      <div className="panel-header">
        <div className="symbol-info">
          <h2>{symbol}</h2>
          <span className="timeframe">{timeframe}</span>
          {data.length > 0 && (
            <span className={`price ${data[data.length - 1].close >= data[data.length - 1].open ? 'positive' : 'negative'}`}>
              ${data[data.length - 1].close.toFixed(2)}
            </span>
          )}
        </div>
        <Tabs tabs={tabs} activeTab={activeTab} onTabChange={setActiveTab} />
      </div>

      <div className="panel-content">
        {activeTab === 'chart' && (
          <CandlestickChart
            data={data}
            signals={signals}
            height={500}
            showVolume={true}
            theme="dark"
          />
        )}

        {activeTab === 'signals' && (
          <SignalsList signals={signals || []} />
        )}

        {activeTab === 'info' && (
          <div className="chart-info">
            <div className="info-grid">
              {data.length > 0 && (
                <>
                  <div className="info-item">
                    <span className="label">Open</span>
                    <span className="value">${data[data.length - 1].open.toFixed(2)}</span>
                  </div>
                  <div className="info-item">
                    <span className="label">High</span>
                    <span className="value">${data[data.length - 1].high.toFixed(2)}</span>
                  </div>
                  <div className="info-item">
                    <span className="label">Low</span>
                    <span className="value">${data[data.length - 1].low.toFixed(2)}</span>
                  </div>
                  <div className="info-item">
                    <span className="label">Close</span>
                    <span className="value">${data[data.length - 1].close.toFixed(2)}</span>
                  </div>
                  <div className="info-item">
                    <span className="label">Volume</span>
                    <span className="value">{data[data.length - 1].volume.toLocaleString()}</span>
                  </div>
                  <div className="info-item">
                    <span className="label">Bars</span>
                    <span className="value">{data.length}</span>
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

// =============================================================================
// WATCHLIST
// =============================================================================

interface WatchlistItem {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  sparkline?: number[];
}

interface WatchlistProps {
  items: WatchlistItem[];
  onItemClick?: (symbol: string) => void;
  selectedSymbol?: string;
}

export const Watchlist: React.FC<WatchlistProps> = ({
  items,
  onItemClick,
  selectedSymbol,
}) => {
  return (
    <Card title="Watchlist" className="watchlist">
      <div className="watchlist-items">
        {items.map((item) => (
          <div
            key={item.symbol}
            className={`watchlist-item ${selectedSymbol === item.symbol ? 'selected' : ''}`}
            onClick={() => onItemClick?.(item.symbol)}
          >
            <div className="item-left">
              <span className="symbol">{item.symbol}</span>
              <span className="price">${item.price.toFixed(2)}</span>
            </div>
            <div className="item-right">
              {item.sparkline && (
                <MiniChart data={item.sparkline} width={60} height={24} />
              )}
              <span className={`change ${item.change >= 0 ? 'positive' : 'negative'}`}>
                {item.change >= 0 ? '+' : ''}{item.changePercent.toFixed(2)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};

// =============================================================================
// SIDEBAR
// =============================================================================

interface SidebarProps {
  collapsed?: boolean;
  onToggle?: () => void;
  activeSection: string;
  onSectionChange: (section: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  collapsed = false,
  onToggle,
  activeSection,
  onSectionChange,
}) => {
  const menuItems = [
    { id: 'dashboard', icon: '📊', label: 'Dashboard' },
    { id: 'backtest', icon: '🔬', label: 'Backtest' },
    { id: 'signals', icon: '📡', label: 'Signals' },
    { id: 'ml', icon: '🤖', label: 'ML Discovery' },
    { id: 'portfolio', icon: '💼', label: 'Portfolio' },
    { id: 'reports', icon: '📄', label: 'Reports' },
    { id: 'settings', icon: '⚙️', label: 'Settings' },
  ];

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <div className="logo">
          {collapsed ? '⍺' : '⍺ AlphaTerminal'}
        </div>
        <button className="toggle-btn" onClick={onToggle}>
          {collapsed ? '→' : '←'}
        </button>
      </div>

      <nav className="sidebar-nav">
        {menuItems.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${activeSection === item.id ? 'active' : ''}`}
            onClick={() => onSectionChange(item.id)}
            title={collapsed ? item.label : undefined}
          >
            <span className="nav-icon">{item.icon}</span>
            {!collapsed && <span className="nav-label">{item.label}</span>}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        {!collapsed && (
          <div className="version">v1.0.0</div>
        )}
      </div>
    </aside>
  );
};

// =============================================================================
// MAIN DASHBOARD LAYOUT
// =============================================================================

interface DashboardLayoutProps {
  children: React.ReactNode;
  sidebar?: React.ReactNode;
  header?: React.ReactNode;
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  children,
  sidebar,
  header,
}) => {
  return (
    <div className="dashboard-layout">
      {sidebar}
      <div className="main-content">
        {header}
        <main className="content-area">
          {children}
        </main>
      </div>
    </div>
  );
};

export default DashboardLayout;
