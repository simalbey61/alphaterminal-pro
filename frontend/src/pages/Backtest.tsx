import React, { useState } from 'react';
import { Play, Calendar, TrendingUp, BarChart3 } from 'lucide-react';

export default function Backtest() {
  const [running, setRunning] = useState(false);
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Backtest</h1>
        <button 
          onClick={() => setRunning(!running)}
          className="px-4 py-2 bg-green-600 rounded-lg hover:bg-green-700 flex items-center space-x-2"
        >
          <Play className="w-4 h-4" />
          <span>{running ? 'Durdur' : 'Başlat'}</span>
        </button>
      </div>
      
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <p className="text-surface-400 text-sm">Strateji</p>
          <select className="mt-2 w-full bg-surface-800 border border-surface-700 rounded-lg px-3 py-2">
            <option>SMC + OrderFlow</option>
            <option>Alpha Momentum</option>
          </select>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <p className="text-surface-400 text-sm">Başlangıç</p>
          <input type="date" className="mt-2 w-full bg-surface-800 border border-surface-700 rounded-lg px-3 py-2" />
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <p className="text-surface-400 text-sm">Bitiş</p>
          <input type="date" className="mt-2 w-full bg-surface-800 border border-surface-700 rounded-lg px-3 py-2" />
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
          <p className="text-surface-400 text-sm">Sermaye</p>
          <input type="number" defaultValue={100000} className="mt-2 w-full bg-surface-800 border border-surface-700 rounded-lg px-3 py-2" />
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
          <TrendingUp className="w-8 h-8 text-green-500 mb-2" />
          <p className="text-surface-400 text-sm">Net Profit</p>
          <p className="text-2xl font-bold text-green-400">+32.5%</p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
          <p className="text-surface-400 text-sm">Win Rate</p>
          <p className="text-2xl font-bold text-white">68.5%</p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
          <p className="text-surface-400 text-sm">Profit Factor</p>
          <p className="text-2xl font-bold text-white">2.15</p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
          <p className="text-surface-400 text-sm">Max Drawdown</p>
          <p className="text-2xl font-bold text-red-400">-8.5%</p>
        </div>
      </div>

      <div className="bg-surface-900 rounded-xl border border-surface-800 p-6 h-[400px] flex items-center justify-center">
        <div className="text-center text-surface-500">
          <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p>Backtest başlatın sonuçları görmek için</p>
        </div>
      </div>
    </div>
  );
}
