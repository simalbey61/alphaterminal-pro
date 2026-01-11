import React from 'react';
import { Brain, Play, Pause, Settings, TrendingUp } from 'lucide-react';

const strategies = [
  { id: 1, name: 'SMC + OrderFlow', status: 'active', winRate: 72.5, trades: 156, pnl: 45200 },
  { id: 2, name: 'Alpha Momentum', status: 'active', winRate: 68.2, trades: 98, pnl: 32100 },
  { id: 3, name: 'Mean Reversion', status: 'paused', winRate: 55.8, trades: 45, pnl: 8500 },
];

export default function Strategies() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Stratejiler</h1>
        <button className="px-4 py-2 bg-primary-600 rounded-lg hover:bg-primary-700 flex items-center space-x-2">
          <Brain className="w-4 h-4" />
          <span>Yeni Strateji</span>
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {strategies.map((s) => (
          <div key={s.id} className="bg-surface-900 rounded-xl border border-surface-800 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-white">{s.name}</h3>
              <span className={`px-2 py-1 rounded text-xs ${
                s.status === 'active' ? 'bg-green-500/20 text-green-400' : 'bg-surface-700 text-surface-400'
              }`}>
                {s.status === 'active' ? 'Aktif' : 'Durduruldu'}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-surface-500 text-xs">Win Rate</p>
                <p className="text-white font-medium">{s.winRate}%</p>
              </div>
              <div>
                <p className="text-surface-500 text-xs">Trade</p>
                <p className="text-white font-medium">{s.trades}</p>
              </div>
            </div>
            <div className="flex items-center justify-between pt-4 border-t border-surface-800">
              <span className="text-green-400 font-medium">+₺{s.pnl.toLocaleString()}</span>
              <div className="flex space-x-2">
                <button className="p-2 bg-surface-800 rounded-lg hover:bg-surface-700">
                  {s.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                <button className="p-2 bg-surface-800 rounded-lg hover:bg-surface-700">
                  <Settings className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
