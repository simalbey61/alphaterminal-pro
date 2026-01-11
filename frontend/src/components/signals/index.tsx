/**
 * AlphaTerminal Pro - Signal Components
 * ======================================
 * 
 * Trading signal display and management.
 */

import React, { useState } from 'react';
import { Card, Badge, Button, EmptyState } from '../common';
import type { TradingSignal, SignalType } from '../../types';

// =============================================================================
// SIGNAL CARD
// =============================================================================

interface SignalCardProps {
  signal: TradingSignal;
  onAction?: (action: 'accept' | 'reject' | 'details') => void;
}

export const SignalCard: React.FC<SignalCardProps> = ({ signal, onAction }) => {
  const isBuy = signal.signalType === 'buy' as SignalType;
  
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'danger';
  };

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const rrRatio = signal.targetPrice && signal.stopLoss
    ? Math.abs(signal.targetPrice - signal.price) / Math.abs(signal.price - signal.stopLoss)
    : null;

  return (
    <div className={`signal-card ${isBuy ? 'buy' : 'sell'}`}>
      <div className="signal-header">
        <div className="signal-type">
          <span className="signal-icon">{isBuy ? '🟢' : '🔴'}</span>
          <span className="signal-label">{signal.signalType.toUpperCase()}</span>
        </div>
        <Badge variant={getConfidenceColor(signal.confidence)} size="sm">
          {(signal.confidence * 100).toFixed(0)}%
        </Badge>
      </div>

      <div className="signal-symbol">
        <h3>{signal.symbol}</h3>
        <span className="timeframe">{signal.timeframe}</span>
      </div>

      <div className="signal-prices">
        <div className="price-row">
          <span className="label">Entry</span>
          <span className="value">${signal.price.toFixed(2)}</span>
        </div>
        {signal.targetPrice && (
          <div className="price-row target">
            <span className="label">Target</span>
            <span className="value">${signal.targetPrice.toFixed(2)}</span>
          </div>
        )}
        {signal.stopLoss && (
          <div className="price-row stop">
            <span className="label">Stop Loss</span>
            <span className="value">${signal.stopLoss.toFixed(2)}</span>
          </div>
        )}
        {rrRatio && (
          <div className="price-row rr">
            <span className="label">R:R Ratio</span>
            <span className="value">{rrRatio.toFixed(2)}</span>
          </div>
        )}
      </div>

      <div className="signal-meta">
        <span className="strategy">{signal.strategy}</span>
        <span className="time">{formatTime(signal.createdAt)}</span>
      </div>

      {signal.indicators && Object.keys(signal.indicators).length > 0 && (
        <div className="signal-indicators">
          {Object.entries(signal.indicators).slice(0, 3).map(([key, value]) => (
            <span key={key} className="indicator">
              {key}: {typeof value === 'number' ? value.toFixed(2) : value}
            </span>
          ))}
        </div>
      )}

      <div className="signal-actions">
        <Button size="sm" variant="success" onClick={() => onAction?.('accept')}>
          Accept
        </Button>
        <Button size="sm" variant="ghost" onClick={() => onAction?.('details')}>
          Details
        </Button>
        <Button size="sm" variant="danger" onClick={() => onAction?.('reject')}>
          Reject
        </Button>
      </div>
    </div>
  );
};

// =============================================================================
// SIGNALS LIST
// =============================================================================

interface SignalsListProps {
  signals: TradingSignal[];
  onSignalAction?: (signal: TradingSignal, action: string) => void;
  loading?: boolean;
}

export const SignalsList: React.FC<SignalsListProps> = ({
  signals,
  onSignalAction,
  loading,
}) => {
  const [filter, setFilter] = useState<'all' | 'buy' | 'sell'>('all');
  const [sortBy, setSortBy] = useState<'time' | 'confidence'>('time');

  const filteredSignals = signals
    .filter(s => filter === 'all' || s.signalType === filter)
    .sort((a, b) => {
      if (sortBy === 'confidence') {
        return b.confidence - a.confidence;
      }
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    });

  if (loading) {
    return <div className="signals-loading">Loading signals...</div>;
  }

  if (signals.length === 0) {
    return (
      <EmptyState
        icon="📡"
        title="No Active Signals"
        description="No trading signals have been generated yet."
      />
    );
  }

  return (
    <div className="signals-list">
      <div className="signals-toolbar">
        <div className="filter-buttons">
          <button
            className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            All ({signals.length})
          </button>
          <button
            className={`filter-btn buy ${filter === 'buy' ? 'active' : ''}`}
            onClick={() => setFilter('buy')}
          >
            Buy ({signals.filter(s => s.signalType === 'buy').length})
          </button>
          <button
            className={`filter-btn sell ${filter === 'sell' ? 'active' : ''}`}
            onClick={() => setFilter('sell')}
          >
            Sell ({signals.filter(s => s.signalType === 'sell').length})
          </button>
        </div>
        <div className="sort-select">
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value as 'time' | 'confidence')}>
            <option value="time">Latest First</option>
            <option value="confidence">Highest Confidence</option>
          </select>
        </div>
      </div>

      <div className="signals-grid">
        {filteredSignals.map((signal) => (
          <SignalCard
            key={signal.id}
            signal={signal}
            onAction={(action) => onSignalAction?.(signal, action)}
          />
        ))}
      </div>
    </div>
  );
};

// =============================================================================
// SIGNAL DETAIL MODAL
// =============================================================================

interface SignalDetailProps {
  signal: TradingSignal | null;
  onClose: () => void;
  onAccept?: () => void;
  onReject?: () => void;
}

export const SignalDetail: React.FC<SignalDetailProps> = ({
  signal,
  onClose,
  onAccept,
  onReject,
}) => {
  if (!signal) return null;

  const isBuy = signal.signalType === 'buy';

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal signal-detail-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Signal Details</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="modal-body">
          <div className={`signal-type-banner ${isBuy ? 'buy' : 'sell'}`}>
            <span className="icon">{isBuy ? '🟢' : '🔴'}</span>
            <span className="type">{signal.signalType.toUpperCase()}</span>
            <span className="symbol">{signal.symbol}</span>
          </div>

          <div className="detail-section">
            <h3>Price Levels</h3>
            <div className="detail-grid">
              <div className="detail-item">
                <span className="label">Entry Price</span>
                <span className="value">${signal.price.toFixed(2)}</span>
              </div>
              {signal.targetPrice && (
                <div className="detail-item positive">
                  <span className="label">Target Price</span>
                  <span className="value">${signal.targetPrice.toFixed(2)}</span>
                  <span className="percent">
                    +{(((signal.targetPrice - signal.price) / signal.price) * 100).toFixed(2)}%
                  </span>
                </div>
              )}
              {signal.stopLoss && (
                <div className="detail-item negative">
                  <span className="label">Stop Loss</span>
                  <span className="value">${signal.stopLoss.toFixed(2)}</span>
                  <span className="percent">
                    {(((signal.stopLoss - signal.price) / signal.price) * 100).toFixed(2)}%
                  </span>
                </div>
              )}
            </div>
          </div>

          <div className="detail-section">
            <h3>Signal Info</h3>
            <div className="detail-grid">
              <div className="detail-item">
                <span className="label">Strategy</span>
                <span className="value">{signal.strategy}</span>
              </div>
              <div className="detail-item">
                <span className="label">Timeframe</span>
                <span className="value">{signal.timeframe}</span>
              </div>
              <div className="detail-item">
                <span className="label">Confidence</span>
                <span className="value">{(signal.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="detail-item">
                <span className="label">Generated</span>
                <span className="value">
                  {new Date(signal.createdAt).toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          {signal.indicators && Object.keys(signal.indicators).length > 0 && (
            <div className="detail-section">
              <h3>Indicators</h3>
              <div className="indicators-grid">
                {Object.entries(signal.indicators).map(([key, value]) => (
                  <div key={key} className="indicator-item">
                    <span className="name">{key}</span>
                    <span className="value">
                      {typeof value === 'number' ? value.toFixed(2) : value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {signal.notes && (
            <div className="detail-section">
              <h3>Notes</h3>
              <p className="notes-text">{signal.notes}</p>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <Button variant="danger" onClick={onReject}>
            Reject Signal
          </Button>
          <Button variant="success" onClick={onAccept}>
            Accept & Trade
          </Button>
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// SIGNALS SUMMARY
// =============================================================================

interface SignalsSummaryProps {
  signals: TradingSignal[];
}

export const SignalsSummary: React.FC<SignalsSummaryProps> = ({ signals }) => {
  const buySignals = signals.filter(s => s.signalType === 'buy');
  const sellSignals = signals.filter(s => s.signalType === 'sell');
  const avgConfidence = signals.length > 0
    ? signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length
    : 0;

  const highConfidence = signals.filter(s => s.confidence >= 0.8).length;

  return (
    <Card title="Signals Summary" className="signals-summary">
      <div className="summary-grid">
        <div className="summary-item">
          <span className="summary-value">{signals.length}</span>
          <span className="summary-label">Total Signals</span>
        </div>
        <div className="summary-item buy">
          <span className="summary-value">{buySignals.length}</span>
          <span className="summary-label">Buy Signals</span>
        </div>
        <div className="summary-item sell">
          <span className="summary-value">{sellSignals.length}</span>
          <span className="summary-label">Sell Signals</span>
        </div>
        <div className="summary-item">
          <span className="summary-value">{(avgConfidence * 100).toFixed(0)}%</span>
          <span className="summary-label">Avg Confidence</span>
        </div>
        <div className="summary-item highlight">
          <span className="summary-value">{highConfidence}</span>
          <span className="summary-label">High Confidence</span>
        </div>
      </div>
    </Card>
  );
};

export default SignalsList;
