import React, { useState } from 'react';
import {
  Filter,
  Search,
  TrendingUp,
  TrendingDown,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Eye,
  Play,
  Pause,
  RefreshCw,
} from 'lucide-react';
import { useStore } from '../store/useStore';
import SignalCard from '../components/dashboard/SignalCard';

// Mock signals data
const mockSignals = [
  {
    id: 'SIG_001',
    symbol: 'THYAO',
    direction: 'LONG' as const,
    strength: 'VERY_STRONG' as const,
    entryPrice: 285.50,
    stopLoss: 275.00,
    takeProfit1: 295.00,
    takeProfit2: 310.00,
    takeProfit3: 325.00,
    riskReward: 2.73,
    confidence: 85,
    smcContext: 'Bullish BOS | OB test | Premium zone',
    orderflowContext: 'Delta +2.5M | Institutional buying',
    alphaContext: 'RS positive | Outperformer',
    timeframe: '4H',
    createdAt: new Date().toISOString(),
    status: 'ACTIVE' as const,
  },
  {
    id: 'SIG_002',
    symbol: 'GARAN',
    direction: 'LONG' as const,
    strength: 'STRONG' as const,
    entryPrice: 52.80,
    stopLoss: 50.50,
    takeProfit1: 55.00,
    takeProfit2: 58.00,
    takeProfit3: 62.00,
    riskReward: 2.15,
    confidence: 72,
    smcContext: 'CHoCH detected | FVG fill',
    orderflowContext: 'CVD divergence | Whale entry',
    alphaContext: 'Sector leader | Beta 1.2',
    timeframe: '1H',
    createdAt: new Date(Date.now() - 3600000).toISOString(),
    status: 'ACTIVE' as const,
  },
  {
    id: 'SIG_003',
    symbol: 'TCELL',
    direction: 'SHORT' as const,
    strength: 'MODERATE' as const,
    entryPrice: 78.50,
    stopLoss: 82.00,
    takeProfit1: 75.00,
    takeProfit2: 72.00,
    takeProfit3: 68.00,
    riskReward: 1.85,
    confidence: 65,
    smcContext: 'Bearish structure | Supply zone',
    orderflowContext: 'Delta negative | Distribution',
    alphaContext: 'Underperformer | RS negative',
    timeframe: '4H',
    createdAt: new Date(Date.now() - 7200000).toISOString(),
    status: 'TRIGGERED' as const,
  },
];

export default function Signals() {
  const [filter, setFilter] = useState<'all' | 'active' | 'triggered' | 'expired'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSignal, setSelectedSignal] = useState<string | null>(null);

  const filteredSignals = mockSignals.filter((signal) => {
    if (filter !== 'all' && signal.status.toLowerCase() !== filter) return false;
    if (searchTerm && !signal.symbol.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ACTIVE':
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      case 'TRIGGERED':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'EXPIRED':
        return <XCircle className="w-4 h-4 text-surface-500" />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Sinyaller</h1>
          <p className="text-surface-400 mt-1">
            Multi-engine sinyal sistemi • Gerçek zamanlı güncellemeler
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button className="flex items-center space-x-2 px-4 py-2 bg-surface-800 rounded-lg hover:bg-surface-700 transition-colors">
            <RefreshCw className="w-4 h-4" />
            <span>Yenile</span>
          </button>
          <button className="flex items-center space-x-2 px-4 py-2 bg-primary-600 rounded-lg hover:bg-primary-700 transition-colors">
            <Play className="w-4 h-4" />
            <span>Tarama Başlat</span>
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <div className="flex items-center justify-between">
            <span className="text-surface-400 text-sm">Aktif Sinyaller</span>
            <AlertCircle className="w-5 h-5 text-yellow-500" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {mockSignals.filter((s) => s.status === 'ACTIVE').length}
          </p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <div className="flex items-center justify-between">
            <span className="text-surface-400 text-sm">Bugün Tetiklenen</span>
            <CheckCircle className="w-5 h-5 text-green-500" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">12</p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <div className="flex items-center justify-between">
            <span className="text-surface-400 text-sm">Win Rate</span>
            <TrendingUp className="w-5 h-5 text-green-500" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">72.5%</p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <div className="flex items-center justify-between">
            <span className="text-surface-400 text-sm">Ort. R:R</span>
            <TrendingUp className="w-5 h-5 text-primary-500" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">2.35</p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center justify-between bg-surface-900 rounded-xl border border-surface-800 p-4">
        <div className="flex items-center space-x-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-surface-500" />
            <input
              type="text"
              placeholder="Sembol ara..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 bg-surface-800 border border-surface-700 rounded-lg text-sm w-64 focus:outline-none focus:border-primary-500"
            />
          </div>
          <div className="flex items-center space-x-2">
            {(['all', 'active', 'triggered', 'expired'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filter === f
                    ? 'bg-primary-600 text-white'
                    : 'bg-surface-800 text-surface-400 hover:text-white'
                }`}
              >
                {f === 'all' ? 'Tümü' : f === 'active' ? 'Aktif' : f === 'triggered' ? 'Tetiklenen' : 'Süresi Dolan'}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <select className="bg-surface-800 border border-surface-700 rounded-lg px-3 py-2 text-sm">
            <option>Timeframe: Tümü</option>
            <option>1H</option>
            <option>4H</option>
            <option>1D</option>
          </select>
          <select className="bg-surface-800 border border-surface-700 rounded-lg px-3 py-2 text-sm">
            <option>Güç: Tümü</option>
            <option>Çok Güçlü</option>
            <option>Güçlü</option>
            <option>Orta</option>
          </select>
        </div>
      </div>

      {/* Signal List */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filteredSignals.map((signal) => (
          <div
            key={signal.id}
            className={`bg-surface-900 rounded-xl border transition-all cursor-pointer ${
              selectedSignal === signal.id
                ? 'border-primary-500 ring-1 ring-primary-500/20'
                : 'border-surface-800 hover:border-surface-700'
            }`}
            onClick={() => setSelectedSignal(selectedSignal === signal.id ? null : signal.id)}
          >
            {/* Signal Header */}
            <div className="p-4 border-b border-surface-800">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div
                    className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                      signal.direction === 'LONG' ? 'bg-green-500/20' : 'bg-red-500/20'
                    }`}
                  >
                    {signal.direction === 'LONG' ? (
                      <TrendingUp className="w-5 h-5 text-green-500" />
                    ) : (
                      <TrendingDown className="w-5 h-5 text-red-500" />
                    )}
                  </div>
                  <div>
                    <h3 className="font-bold text-white">{signal.symbol}</h3>
                    <p className="text-sm text-surface-400">{signal.timeframe} • {signal.direction}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {getStatusIcon(signal.status)}
                  <span
                    className={`px-2 py-1 rounded-full text-xs font-medium ${
                      signal.strength === 'VERY_STRONG'
                        ? 'bg-green-500/20 text-green-400'
                        : signal.strength === 'STRONG'
                        ? 'bg-blue-500/20 text-blue-400'
                        : 'bg-yellow-500/20 text-yellow-400'
                    }`}
                  >
                    {signal.strength === 'VERY_STRONG' ? 'Çok Güçlü' : signal.strength === 'STRONG' ? 'Güçlü' : 'Orta'}
                  </span>
                </div>
              </div>
            </div>

            {/* Signal Body */}
            <div className="p-4 grid grid-cols-3 gap-4">
              <div>
                <p className="text-xs text-surface-500 mb-1">Entry</p>
                <p className="font-mono font-medium text-white">₺{signal.entryPrice.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-xs text-surface-500 mb-1">Stop Loss</p>
                <p className="font-mono font-medium text-red-400">₺{signal.stopLoss.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-xs text-surface-500 mb-1">TP1</p>
                <p className="font-mono font-medium text-green-400">₺{signal.takeProfit1.toFixed(2)}</p>
              </div>
            </div>

            {/* Confidence Bar */}
            <div className="px-4 pb-4">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-surface-500">Güven</span>
                <span className="text-white font-medium">{signal.confidence}%</span>
              </div>
              <div className="h-2 bg-surface-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    signal.confidence >= 80
                      ? 'bg-green-500'
                      : signal.confidence >= 60
                      ? 'bg-blue-500'
                      : 'bg-yellow-500'
                  }`}
                  style={{ width: `${signal.confidence}%` }}
                />
              </div>
            </div>

            {/* Expanded Details */}
            {selectedSignal === signal.id && (
              <div className="px-4 pb-4 border-t border-surface-800 pt-4 animate-fade-in">
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-surface-500 mb-1">SMC Context</p>
                    <p className="text-sm text-surface-300">{signal.smcContext}</p>
                  </div>
                  <div>
                    <p className="text-xs text-surface-500 mb-1">OrderFlow Context</p>
                    <p className="text-sm text-surface-300">{signal.orderflowContext}</p>
                  </div>
                  <div>
                    <p className="text-xs text-surface-500 mb-1">Alpha Context</p>
                    <p className="text-sm text-surface-300">{signal.alphaContext}</p>
                  </div>
                  <div className="grid grid-cols-3 gap-4 pt-3">
                    <div>
                      <p className="text-xs text-surface-500 mb-1">TP2</p>
                      <p className="font-mono text-green-400">₺{signal.takeProfit2.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-surface-500 mb-1">TP3</p>
                      <p className="font-mono text-green-400">₺{signal.takeProfit3.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-surface-500 mb-1">R:R</p>
                      <p className="font-mono text-primary-400">1:{signal.riskReward.toFixed(2)}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2 pt-3">
                    <button className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-medium transition-colors">
                      Execute Trade
                    </button>
                    <button className="px-4 py-2 bg-surface-800 hover:bg-surface-700 rounded-lg text-sm font-medium transition-colors">
                      <Eye className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
