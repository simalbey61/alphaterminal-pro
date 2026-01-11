import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  Target,
  AlertTriangle,
  Clock,
  Layers,
  Zap,
  Brain,
  DollarSign,
} from 'lucide-react';
import TradingChart from '../components/charts/TradingChart';
import SMCPanel from '../components/analysis/SMCPanel';
import OrderFlowPanel from '../components/analysis/OrderFlowPanel';
import AlphaPanel from '../components/analysis/AlphaPanel';

// Tab types
type TabType = 'overview' | 'smc' | 'orderflow' | 'alpha' | 'risk';

export default function Analysis() {
  const { symbol } = useParams<{ symbol: string }>();
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [selectedSymbol, setSelectedSymbol] = useState(symbol || 'THYAO');
  const [timeframe, setTimeframe] = useState('4H');

  // Mock analysis data
  const analysisData = {
    symbol: selectedSymbol,
    price: 285.50,
    change: 2.35,
    changePercent: 0.83,
    volume: 15420000,
    avgVolume: 12500000,
    
    // SMC Data
    smc: {
      structure: 'BULLISH',
      structureLabel: 'Higher Highs, Higher Lows',
      bias: 'LONG',
      bos: true,
      choch: false,
      activeOB: { type: 'BULLISH', top: 290, bottom: 282 },
      activeFVG: { type: 'BULLISH', top: 288, bottom: 285 },
      liquiditySweep: 'YOK',
      confluenceScore: 78,
    },
    
    // OrderFlow Data
    orderflow: {
      flowBias: 'BULLISH',
      flowScore: 72,
      delta: { value: 2500000, trend: 'POSITIVE' },
      cvd: { value: 15000000, trend: 'RISING' },
      vwap: { value: 283.50, position: 'ABOVE' },
      institutionalBuying: true,
      whaleActivity: true,
    },
    
    // Alpha Data
    alpha: {
      alphaScore: 75,
      category: 'OUTPERFORMER',
      jensensAlpha: 0.15,
      beta: 1.25,
      sharpe: 1.85,
      sortino: 2.12,
      maxDrawdown: 8.5,
      rsSlope: 'POSITIVE',
      momentum: 'STRONG',
    },
    
    // Risk Data
    risk: {
      volatility: 28.5,
      var95: 4.2,
      atr: 5.80,
      riskLevel: 'MODERATE',
    },
  };

  const tabs = [
    { id: 'overview', label: 'Özet', icon: Activity },
    { id: 'smc', label: 'SMC', icon: Layers },
    { id: 'orderflow', label: 'OrderFlow', icon: BarChart3 },
    { id: 'alpha', label: 'Alpha', icon: Zap },
    { id: 'risk', label: 'Risk', icon: AlertTriangle },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div>
            <div className="flex items-center space-x-3">
              <h1 className="text-2xl font-bold text-white">{selectedSymbol}</h1>
              <span className={`px-2 py-1 rounded-lg text-sm font-medium ${
                analysisData.change >= 0 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
              }`}>
                {analysisData.change >= 0 ? '+' : ''}{analysisData.changePercent.toFixed(2)}%
              </span>
            </div>
            <p className="text-surface-400 mt-1">Türk Hava Yolları • BIST 30</p>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          {/* Symbol Search */}
          <input
            type="text"
            placeholder="Sembol ara..."
            className="px-4 py-2 bg-surface-800 border border-surface-700 rounded-lg text-sm w-40"
            onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
          />
          
          {/* Timeframe Selector */}
          <div className="flex items-center bg-surface-800 rounded-lg p-1">
            {['1H', '4H', '1D', '1W'].map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  timeframe === tf
                    ? 'bg-primary-600 text-white'
                    : 'text-surface-400 hover:text-white'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Price Display */}
      <div className="grid grid-cols-5 gap-4">
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <p className="text-surface-400 text-sm mb-1">Fiyat</p>
          <p className="text-2xl font-bold text-white">₺{analysisData.price.toFixed(2)}</p>
          <p className={`text-sm mt-1 ${analysisData.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {analysisData.change >= 0 ? '+' : ''}₺{analysisData.change.toFixed(2)}
          </p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <p className="text-surface-400 text-sm mb-1">Hacim</p>
          <p className="text-2xl font-bold text-white">{(analysisData.volume / 1000000).toFixed(1)}M</p>
          <p className="text-sm mt-1 text-surface-500">Ort: {(analysisData.avgVolume / 1000000).toFixed(1)}M</p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <p className="text-surface-400 text-sm mb-1">SMC Bias</p>
          <p className={`text-2xl font-bold ${analysisData.smc.bias === 'LONG' ? 'text-green-400' : 'text-red-400'}`}>
            {analysisData.smc.bias}
          </p>
          <p className="text-sm mt-1 text-surface-500">Skor: {analysisData.smc.confluenceScore}/100</p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <p className="text-surface-400 text-sm mb-1">Alpha Score</p>
          <p className="text-2xl font-bold text-primary-400">{analysisData.alpha.alphaScore}/100</p>
          <p className="text-sm mt-1 text-surface-500">{analysisData.alpha.category}</p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <p className="text-surface-400 text-sm mb-1">Risk Level</p>
          <p className="text-2xl font-bold text-yellow-400">{analysisData.risk.riskLevel}</p>
          <p className="text-sm mt-1 text-surface-500">ATR: {analysisData.risk.atr.toFixed(2)}</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center space-x-1 bg-surface-900 rounded-xl border border-surface-800 p-1">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as TabType)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-primary-600 text-white'
                  : 'text-surface-400 hover:text-white hover:bg-surface-800'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart */}
        <div className="lg:col-span-2 bg-surface-900 rounded-xl border border-surface-800">
          <div className="p-4 border-b border-surface-800">
            <h2 className="font-semibold text-white">Grafik</h2>
          </div>
          <div className="p-4 h-[500px]">
            <TradingChart symbol={selectedSymbol} timeframe={timeframe} />
          </div>
        </div>

        {/* Analysis Panel */}
        <div className="bg-surface-900 rounded-xl border border-surface-800">
          {activeTab === 'overview' && (
            <div className="p-4">
              <h2 className="font-semibold text-white mb-4">Analiz Özeti</h2>
              
              {/* Structure */}
              <div className="mb-6">
                <h3 className="text-sm text-surface-400 mb-2">Market Yapısı</h3>
                <div className="flex items-center space-x-2">
                  <span className={`px-3 py-1 rounded-lg text-sm font-medium ${
                    analysisData.smc.structure === 'BULLISH'
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-red-500/20 text-red-400'
                  }`}>
                    {analysisData.smc.structure}
                  </span>
                  <span className="text-surface-500 text-sm">{analysisData.smc.structureLabel}</span>
                </div>
              </div>

              {/* Key Levels */}
              <div className="mb-6">
                <h3 className="text-sm text-surface-400 mb-2">Önemli Seviyeler</h3>
                <div className="space-y-2">
                  {analysisData.smc.activeOB && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-surface-500">Order Block</span>
                      <span className="font-mono text-white">
                        {analysisData.smc.activeOB.bottom} - {analysisData.smc.activeOB.top}
                      </span>
                    </div>
                  )}
                  {analysisData.smc.activeFVG && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-surface-500">FVG</span>
                      <span className="font-mono text-white">
                        {analysisData.smc.activeFVG.bottom} - {analysisData.smc.activeFVG.top}
                      </span>
                    </div>
                  )}
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-surface-500">VWAP</span>
                    <span className="font-mono text-white">₺{analysisData.orderflow.vwap.value}</span>
                  </div>
                </div>
              </div>

              {/* Indicators */}
              <div className="mb-6">
                <h3 className="text-sm text-surface-400 mb-2">Göstergeler</h3>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-surface-800 rounded-lg p-3">
                    <p className="text-xs text-surface-500">BOS</p>
                    <p className={`font-medium ${analysisData.smc.bos ? 'text-green-400' : 'text-surface-500'}`}>
                      {analysisData.smc.bos ? 'Tespit' : 'Yok'}
                    </p>
                  </div>
                  <div className="bg-surface-800 rounded-lg p-3">
                    <p className="text-xs text-surface-500">CHoCH</p>
                    <p className={`font-medium ${analysisData.smc.choch ? 'text-yellow-400' : 'text-surface-500'}`}>
                      {analysisData.smc.choch ? 'Tespit' : 'Yok'}
                    </p>
                  </div>
                  <div className="bg-surface-800 rounded-lg p-3">
                    <p className="text-xs text-surface-500">Delta</p>
                    <p className={`font-medium ${analysisData.orderflow.delta.value > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {(analysisData.orderflow.delta.value / 1000000).toFixed(1)}M
                    </p>
                  </div>
                  <div className="bg-surface-800 rounded-lg p-3">
                    <p className="text-xs text-surface-500">Kurumsal</p>
                    <p className={`font-medium ${analysisData.orderflow.institutionalBuying ? 'text-green-400' : 'text-surface-500'}`}>
                      {analysisData.orderflow.institutionalBuying ? 'Alış' : 'Yok'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Score Summary */}
              <div>
                <h3 className="text-sm text-surface-400 mb-2">Skor Özeti</h3>
                <div className="space-y-3">
                  {[
                    { label: 'SMC', score: analysisData.smc.confluenceScore, color: 'green' },
                    { label: 'OrderFlow', score: analysisData.orderflow.flowScore, color: 'blue' },
                    { label: 'Alpha', score: analysisData.alpha.alphaScore, color: 'purple' },
                  ].map((item) => (
                    <div key={item.label}>
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span className="text-surface-400">{item.label}</span>
                        <span className="text-white font-medium">{item.score}/100</span>
                      </div>
                      <div className="h-2 bg-surface-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full bg-${item.color}-500`}
                          style={{ width: `${item.score}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'smc' && <SMCPanel data={analysisData.smc} />}
          {activeTab === 'orderflow' && <OrderFlowPanel data={analysisData.orderflow} />}
          {activeTab === 'alpha' && <AlphaPanel data={analysisData.alpha} />}
          {activeTab === 'risk' && (
            <div className="p-4">
              <h2 className="font-semibold text-white mb-4">Risk Analizi</h2>
              <div className="space-y-4">
                <div className="bg-surface-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-surface-400">Volatilite (Ann.)</span>
                    <span className="text-white font-medium">{analysisData.risk.volatility}%</span>
                  </div>
                  <div className="h-2 bg-surface-700 rounded-full overflow-hidden">
                    <div className="h-full bg-yellow-500" style={{ width: `${analysisData.risk.volatility}%` }} />
                  </div>
                </div>
                <div className="bg-surface-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-surface-400">VaR (95%)</span>
                    <span className="text-red-400 font-medium">-{analysisData.risk.var95}%</span>
                  </div>
                </div>
                <div className="bg-surface-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-surface-400">ATR (14)</span>
                    <span className="text-white font-medium">₺{analysisData.risk.atr}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
