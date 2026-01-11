/**
 * AlphaTerminal Pro - Backtest Components
 * ========================================
 * 
 * Backtest configuration, results, and analysis.
 */

import React, { useState } from 'react';
import { Card, Button, Input, Select, MetricCard, Badge, LoadingState, ErrorState } from '../common';
import { EquityCurveChart, DrawdownChart } from '../charts';
import type { BacktestConfig, BacktestResult, Trade, TimeFrame } from '../../types';

// =============================================================================
// BACKTEST FORM
// =============================================================================

interface BacktestFormProps {
  onSubmit: (config: BacktestConfig) => void;
  loading?: boolean;
  availableStrategies?: { name: string; type: string; params: Record<string, unknown> }[];
}

export const BacktestForm: React.FC<BacktestFormProps> = ({
  onSubmit,
  loading,
  availableStrategies = [],
}) => {
  const [config, setConfig] = useState<Partial<BacktestConfig>>({
    symbol: 'THYAO',
    interval: '1d' as TimeFrame,
    startDate: '2023-01-01',
    endDate: '2024-01-01',
    initialCapital: 100000,
    commission: 0.001,
    slippage: 0.0005,
  });

  const [selectedStrategy, setSelectedStrategy] = useState('sma_crossover');
  const [strategyParams, setStrategyParams] = useState<Record<string, number | string>>({
    fast_period: 10,
    slow_period: 30,
  });

  const timeframeOptions = [
    { value: '1m', label: '1 Minute' },
    { value: '5m', label: '5 Minutes' },
    { value: '15m', label: '15 Minutes' },
    { value: '1h', label: '1 Hour' },
    { value: '4h', label: '4 Hours' },
    { value: '1d', label: '1 Day' },
    { value: '1w', label: '1 Week' },
  ];

  const defaultStrategies = [
    { value: 'sma_crossover', label: 'SMA Crossover' },
    { value: 'rsi_strategy', label: 'RSI Strategy' },
    { value: 'macd_strategy', label: 'MACD Strategy' },
    { value: 'bollinger_bands', label: 'Bollinger Bands' },
    { value: 'breakout', label: 'Breakout Strategy' },
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    onSubmit({
      ...config,
      strategy: {
        name: selectedStrategy,
        type: selectedStrategy,
        params: strategyParams,
      },
    } as BacktestConfig);
  };

  return (
    <Card title="Backtest Configuration" className="backtest-form">
      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          <Input
            label="Symbol"
            value={config.symbol || ''}
            onChange={(e) => setConfig({ ...config, symbol: e.target.value.toUpperCase() })}
            placeholder="THYAO"
          />

          <Select
            label="Timeframe"
            options={timeframeOptions}
            value={config.interval}
            onChange={(val) => setConfig({ ...config, interval: val as TimeFrame })}
          />

          <Input
            label="Start Date"
            type="date"
            value={config.startDate || ''}
            onChange={(e) => setConfig({ ...config, startDate: e.target.value })}
          />

          <Input
            label="End Date"
            type="date"
            value={config.endDate || ''}
            onChange={(e) => setConfig({ ...config, endDate: e.target.value })}
          />

          <Input
            label="Initial Capital"
            type="number"
            value={config.initialCapital || 100000}
            onChange={(e) => setConfig({ ...config, initialCapital: Number(e.target.value) })}
            prefix="$"
          />

          <Input
            label="Commission (%)"
            type="number"
            step="0.001"
            value={(config.commission || 0) * 100}
            onChange={(e) => setConfig({ ...config, commission: Number(e.target.value) / 100 })}
            suffix="%"
          />
        </div>

        <div className="strategy-section">
          <h3>Strategy</h3>
          
          <Select
            label="Strategy Type"
            options={defaultStrategies}
            value={selectedStrategy}
            onChange={setSelectedStrategy}
          />

          <div className="params-grid">
            {selectedStrategy === 'sma_crossover' && (
              <>
                <Input
                  label="Fast Period"
                  type="number"
                  value={strategyParams.fast_period as number}
                  onChange={(e) => setStrategyParams({
                    ...strategyParams,
                    fast_period: Number(e.target.value)
                  })}
                />
                <Input
                  label="Slow Period"
                  type="number"
                  value={strategyParams.slow_period as number}
                  onChange={(e) => setStrategyParams({
                    ...strategyParams,
                    slow_period: Number(e.target.value)
                  })}
                />
              </>
            )}
            {selectedStrategy === 'rsi_strategy' && (
              <>
                <Input
                  label="RSI Period"
                  type="number"
                  value={strategyParams.period as number || 14}
                  onChange={(e) => setStrategyParams({
                    ...strategyParams,
                    period: Number(e.target.value)
                  })}
                />
                <Input
                  label="Overbought"
                  type="number"
                  value={strategyParams.overbought as number || 70}
                  onChange={(e) => setStrategyParams({
                    ...strategyParams,
                    overbought: Number(e.target.value)
                  })}
                />
                <Input
                  label="Oversold"
                  type="number"
                  value={strategyParams.oversold as number || 30}
                  onChange={(e) => setStrategyParams({
                    ...strategyParams,
                    oversold: Number(e.target.value)
                  })}
                />
              </>
            )}
          </div>
        </div>

        <div className="form-actions">
          <Button type="submit" variant="primary" loading={loading}>
            {loading ? 'Running Backtest...' : 'Run Backtest'}
          </Button>
        </div>
      </form>
    </Card>
  );
};

// =============================================================================
// BACKTEST RESULTS
// =============================================================================

interface BacktestResultsProps {
  result: BacktestResult | null;
  loading?: boolean;
  error?: string;
}

export const BacktestResults: React.FC<BacktestResultsProps> = ({
  result,
  loading,
  error,
}) => {
  const [activeTab, setActiveTab] = useState<'summary' | 'trades' | 'charts'>('summary');

  if (loading) return <LoadingState message="Running backtest..." />;
  if (error) return <ErrorState message={error} />;
  if (!result) return null;

  const isPositive = result.totalReturnPct >= 0;

  return (
    <div className="backtest-results">
      <div className="results-header">
        <div className="result-title">
          <h2>Backtest Results</h2>
          <Badge variant={result.status === 'completed' ? 'success' : 'warning'}>
            {result.status}
          </Badge>
        </div>
        <div className="result-return">
          <span className={`return-value ${isPositive ? 'positive' : 'negative'}`}>
            {isPositive ? '+' : ''}{result.totalReturnPct.toFixed(2)}%
          </span>
          <span className="return-label">Total Return</span>
        </div>
      </div>

      <div className="results-tabs">
        <button
          className={activeTab === 'summary' ? 'active' : ''}
          onClick={() => setActiveTab('summary')}
        >
          Summary
        </button>
        <button
          className={activeTab === 'charts' ? 'active' : ''}
          onClick={() => setActiveTab('charts')}
        >
          Charts
        </button>
        <button
          className={activeTab === 'trades' ? 'active' : ''}
          onClick={() => setActiveTab('trades')}
        >
          Trades ({result.totalTrades})
        </button>
      </div>

      {activeTab === 'summary' && (
        <div className="results-summary">
          <div className="metrics-section">
            <h3>Performance</h3>
            <div className="metrics-grid">
              <MetricCard
                label="Total Return"
                value={`${result.totalReturnPct.toFixed(2)}%`}
                change={result.totalReturnPct}
              />
              <MetricCard
                label="Sharpe Ratio"
                value={result.sharpeRatio.toFixed(2)}
              />
              <MetricCard
                label="Sortino Ratio"
                value={result.sortinoRatio.toFixed(2)}
              />
              <MetricCard
                label="Calmar Ratio"
                value={result.calmarRatio.toFixed(2)}
              />
              <MetricCard
                label="Max Drawdown"
                value={`${(result.maxDrawdown * 100).toFixed(2)}%`}
              />
              <MetricCard
                label="Volatility"
                value={`${(result.volatility * 100).toFixed(2)}%`}
              />
            </div>
          </div>

          <div className="metrics-section">
            <h3>Trade Statistics</h3>
            <div className="metrics-grid">
              <MetricCard
                label="Total Trades"
                value={result.totalTrades}
              />
              <MetricCard
                label="Win Rate"
                value={`${(result.winRate * 100).toFixed(1)}%`}
              />
              <MetricCard
                label="Profit Factor"
                value={result.profitFactor.toFixed(2)}
              />
              <MetricCard
                label="Avg Trade"
                value={`$${result.avgTrade.toFixed(2)}`}
              />
              <MetricCard
                label="Avg Winner"
                value={`$${result.avgWinner.toFixed(2)}`}
              />
              <MetricCard
                label="Avg Loser"
                value={`$${result.avgLoser.toFixed(2)}`}
              />
              <MetricCard
                label="Largest Win"
                value={`$${result.largestWinner.toFixed(2)}`}
              />
              <MetricCard
                label="Largest Loss"
                value={`$${result.largestLoser.toFixed(2)}`}
              />
            </div>
          </div>
        </div>
      )}

      {activeTab === 'charts' && (
        <div className="results-charts">
          <Card title="Equity Curve">
            {result.equityCurve && (
              <EquityCurveChart data={result.equityCurve} height={300} theme="dark" />
            )}
          </Card>
          <Card title="Drawdown">
            {result.equityCurve && (
              <DrawdownChart data={result.equityCurve} height={200} theme="dark" />
            )}
          </Card>
        </div>
      )}

      {activeTab === 'trades' && (
        <div className="results-trades">
          <TradesList trades={result.trades || []} />
        </div>
      )}
    </div>
  );
};

// =============================================================================
// TRADES LIST
// =============================================================================

interface TradesListProps {
  trades: Trade[];
}

export const TradesList: React.FC<TradesListProps> = ({ trades }) => {
  if (trades.length === 0) {
    return <div className="no-trades">No trades executed</div>;
  }

  return (
    <div className="trades-table-container">
      <table className="trades-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Entry Time</th>
            <th>Exit Time</th>
            <th>Side</th>
            <th>Entry</th>
            <th>Exit</th>
            <th>Qty</th>
            <th>P&L</th>
            <th>P&L %</th>
            <th>Exit Reason</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((trade, index) => (
            <tr key={trade.id} className={trade.pnl >= 0 ? 'winning' : 'losing'}>
              <td>{index + 1}</td>
              <td>{new Date(trade.entryTime).toLocaleDateString()}</td>
              <td>{new Date(trade.exitTime).toLocaleDateString()}</td>
              <td>
                <Badge variant={trade.side === 'long' ? 'success' : 'danger'}>
                  {trade.side.toUpperCase()}
                </Badge>
              </td>
              <td>${trade.entryPrice.toFixed(2)}</td>
              <td>${trade.exitPrice.toFixed(2)}</td>
              <td>{trade.quantity}</td>
              <td className={trade.pnl >= 0 ? 'positive' : 'negative'}>
                ${trade.pnl.toFixed(2)}
              </td>
              <td className={trade.pnlPercent >= 0 ? 'positive' : 'negative'}>
                {trade.pnlPercent >= 0 ? '+' : ''}{trade.pnlPercent.toFixed(2)}%
              </td>
              <td>{trade.exitReason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// =============================================================================
// BACKTEST HISTORY
// =============================================================================

interface BacktestHistoryProps {
  backtests: BacktestResult[];
  onSelect?: (backtest: BacktestResult) => void;
}

export const BacktestHistory: React.FC<BacktestHistoryProps> = ({
  backtests,
  onSelect,
}) => {
  if (backtests.length === 0) {
    return (
      <Card title="Backtest History">
        <div className="empty-history">
          <p>No backtests yet. Run your first backtest above.</p>
        </div>
      </Card>
    );
  }

  return (
    <Card title="Backtest History" className="backtest-history">
      <div className="history-list">
        {backtests.map((bt) => (
          <div
            key={bt.id}
            className="history-item"
            onClick={() => onSelect?.(bt)}
          >
            <div className="history-info">
              <span className="symbol">{bt.config.symbol}</span>
              <span className="strategy">{bt.config.strategy.name}</span>
              <span className="date">
                {new Date(bt.createdAt).toLocaleDateString()}
              </span>
            </div>
            <div className="history-result">
              <span className={`return ${bt.totalReturnPct >= 0 ? 'positive' : 'negative'}`}>
                {bt.totalReturnPct >= 0 ? '+' : ''}{bt.totalReturnPct.toFixed(2)}%
              </span>
              <Badge variant={bt.status === 'completed' ? 'success' : 'warning'}>
                {bt.status}
              </Badge>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};

export default BacktestForm;
