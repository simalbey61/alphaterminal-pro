import React from 'react';
import { Wallet, TrendingUp, Target } from 'lucide-react';

export default function Portfolio() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Portföy</h1>
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
          <Wallet className="w-8 h-8 text-primary-500 mb-4" />
          <p className="text-surface-400 text-sm">Toplam Değer</p>
          <p className="text-2xl font-bold text-white">₺1,250,000</p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
          <TrendingUp className="w-8 h-8 text-green-500 mb-4" />
          <p className="text-surface-400 text-sm">Toplam PnL</p>
          <p className="text-2xl font-bold text-green-400">+₺87,500</p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
          <Target className="w-8 h-8 text-blue-500 mb-4" />
          <p className="text-surface-400 text-sm">Açık Pozisyon</p>
          <p className="text-2xl font-bold text-white">5</p>
        </div>
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
          <p className="text-surface-400 text-sm">Win Rate</p>
          <p className="text-2xl font-bold text-white">68.5%</p>
        </div>
      </div>
      <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
        <h2 className="font-semibold text-white mb-4">Pozisyonlar</h2>
        <table className="w-full">
          <thead>
            <tr className="border-b border-surface-800 text-left text-sm text-surface-400">
              <th className="pb-3">Sembol</th>
              <th className="pb-3">Yön</th>
              <th className="pb-3">Giriş</th>
              <th className="pb-3">Güncel</th>
              <th className="pb-3 text-right">PnL</th>
            </tr>
          </thead>
          <tbody className="text-white">
            <tr className="border-b border-surface-800/50">
              <td className="py-3 font-medium">THYAO</td>
              <td><span className="px-2 py-1 bg-green-500/20 text-green-400 rounded text-xs">LONG</span></td>
              <td>₺275.50</td>
              <td>₺285.50</td>
              <td className="text-right text-green-400">+₺5,000</td>
            </tr>
            <tr className="border-b border-surface-800/50">
              <td className="py-3 font-medium">GARAN</td>
              <td><span className="px-2 py-1 bg-green-500/20 text-green-400 rounded text-xs">LONG</span></td>
              <td>₺50.80</td>
              <td>₺52.80</td>
              <td className="text-right text-green-400">+₺4,000</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
                <th className="text-right text-sm font-medium text-surface-400 px-6 py-4">Aksiyon</th>
              </tr>
            </thead>
            <tbody>
              {sortedPositions.map((pos) => (
                <tr key={pos.id} className="border-b border-surface-800/50 hover:bg-surface-800/30 transition-colors">
                  <td className="px-6 py-4">
                    <div>
                      <p className="font-medium text-white">{pos.symbol}</p>
                      <p className="text-sm text-surface-500">{pos.name}</p>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      pos.direction === 'LONG' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                    }`}>
                      {pos.direction}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right font-mono text-white">
                    {pos.quantity.toLocaleString('tr-TR')}
                  </td>
                  <td className="px-6 py-4 text-right font-mono text-surface-300">
                    ₺{pos.entryPrice.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 text-right font-mono text-white">
                    ₺{pos.currentPrice.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className={pos.unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}>
                      <p className="font-mono font-medium">
                        {pos.unrealizedPnL >= 0 ? '+' : ''}{showValues ? `₺${pos.unrealizedPnL.toLocaleString('tr-TR')}` : '••••'}
                      </p>
                      <p className="text-xs">{pos.unrealizedPnLPercent >= 0 ? '+' : ''}{pos.unrealizedPnLPercent.toFixed(2)}%</p>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end space-x-2">
                      <div className="w-16 h-2 bg-surface-700 rounded-full overflow-hidden">
                        <div className="h-full bg-primary-500 rounded-full" style={{ width: `${pos.weight}%` }} />
                      </div>
                      <span className="text-sm text-surface-300 w-12 text-right">{pos.weight}%</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end space-x-2">
                      <button className="px-3 py-1 bg-surface-700 hover:bg-surface-600 rounded text-sm transition-colors">Düzenle</button>
                      <button className="px-3 py-1 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded text-sm transition-colors">Kapat</button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {sortedPositions.length === 0 && (
            <div className="p-12 text-center text-surface-500">
              <Target className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Açık pozisyon yok</p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'history' && (
        <div className="bg-surface-900 rounded-xl border border-surface-800 overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-surface-800">
                <th className="text-left text-sm font-medium text-surface-400 px-6 py-4">Tarih</th>
                <th className="text-left text-sm font-medium text-surface-400 px-6 py-4">Sembol</th>
                <th className="text-left text-sm font-medium text-surface-400 px-6 py-4">Yön</th>
                <th className="text-right text-sm font-medium text-surface-400 px-6 py-4">Giriş</th>
                <th className="text-right text-sm font-medium text-surface-400 px-6 py-4">Çıkış</th>
                <th className="text-right text-sm font-medium text-surface-400 px-6 py-4">Süre</th>
                <th className="text-right text-sm font-medium text-surface-400 px-6 py-4">Neden</th>
                <th className="text-right text-sm font-medium text-surface-400 px-6 py-4">PnL</th>
              </tr>
            </thead>
            <tbody>
              {mockTrades.map((trade) => (
                <tr key={trade.id} className="border-b border-surface-800/50 hover:bg-surface-800/30 transition-colors">
                  <td className="px-6 py-4 text-surface-300">{trade.date}</td>
                  <td className="px-6 py-4 font-medium text-white">{trade.symbol}</td>
                  <td className="px-6 py-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      trade.direction === 'LONG' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                    }`}>{trade.direction}</span>
                  </td>
                  <td className="px-6 py-4 text-right font-mono text-surface-300">₺{trade.entryPrice.toFixed(2)}</td>
                  <td className="px-6 py-4 text-right font-mono text-white">₺{trade.exitPrice.toFixed(2)}</td>
                  <td className="px-6 py-4 text-right text-surface-400">{trade.duration}</td>
                  <td className="px-6 py-4 text-right">
                    <span className={`px-2 py-1 rounded text-xs ${
                      trade.exitReason.includes('TP') ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                    }`}>{trade.exitReason}</span>
                  </td>
                  <td className={`px-6 py-4 text-right font-mono font-medium ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {trade.pnl >= 0 ? '+' : ''}{showValues ? `₺${trade.pnl.toLocaleString('tr-TR')}` : '••••'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {activeTab === 'allocation' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
            <h3 className="font-semibold text-white mb-6">Sektör Dağılımı</h3>
            <div className="space-y-4">
              {sectorAllocation.map((item) => (
                <div key={item.sector}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-surface-300">{item.sector}</span>
                    <div className="flex items-center space-x-4">
                      <span className={`text-sm ${item.pnl >= 0 ? 'text-green-400' : item.pnl < 0 ? 'text-red-400' : 'text-surface-500'}`}>
                        {item.pnl !== 0 && (item.pnl >= 0 ? '+' : '')}{showValues ? `₺${item.pnl.toLocaleString()}` : '••••'}
                      </span>
                      <span className="text-white font-medium w-16 text-right">{item.weight}%</span>
                    </div>
                  </div>
                  <div className="h-3 bg-surface-700 rounded-full overflow-hidden">
                    <div className={`h-full ${item.color} rounded-full transition-all`} style={{ width: `${item.weight}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
            <h3 className="font-semibold text-white mb-6">Sembol Dağılımı</h3>
            <div className="space-y-3">
              {mockPositions.map((pos) => (
                <div key={pos.id} className="flex items-center justify-between p-3 bg-surface-800 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${pos.unrealizedPnL >= 0 ? 'bg-green-500' : 'bg-red-500'}`} />
                    <span className="font-medium text-white">{pos.symbol}</span>
                  </div>
                  <div className="text-right">
                    <p className="text-white font-medium">{pos.weight}%</p>
                    <p className={`text-xs ${pos.unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {pos.unrealizedPnL >= 0 ? '+' : ''}{pos.unrealizedPnLPercent.toFixed(2)}%
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'risk' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
            <h3 className="font-semibold text-white mb-6">Risk Metrikleri</h3>
            <div className="space-y-4">
              {[
                { label: 'Max Drawdown', value: `${stats.maxDrawdown}%`, limit: '15%', pct: (stats.maxDrawdown / 15) * 100, color: stats.maxDrawdown > 10 ? 'red' : stats.maxDrawdown > 7 ? 'yellow' : 'green' },
                { label: 'Exposure', value: `${stats.exposure}%`, limit: '80%', pct: (stats.exposure / 80) * 100, color: stats.exposure > 70 ? 'yellow' : 'green' },
                { label: 'Portfolio Beta', value: '1.15', limit: '1.5', pct: 76, color: 'blue' },
                { label: 'VaR (95%)', value: '-3.5%', limit: '-5%', pct: 70, color: 'yellow' },
              ].map((item) => (
                <div key={item.label} className="p-4 bg-surface-800 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-surface-400">{item.label}</span>
                    <span className={`font-medium text-${item.color}-400`}>{item.value}</span>
                  </div>
                  <div className="h-2 bg-surface-700 rounded-full overflow-hidden">
                    <div className={`h-full bg-${item.color}-500 rounded-full`} style={{ width: `${Math.min(item.pct, 100)}%` }} />
                  </div>
                  <p className="text-xs text-surface-500 mt-1">Limit: {item.limit}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
            <h3 className="font-semibold text-white mb-6">Performans Metrikleri</h3>
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: 'Sharpe Ratio', value: stats.sharpeRatio.toFixed(2), good: stats.sharpeRatio >= 1.5 },
                { label: 'Sortino Ratio', value: '2.12', good: true },
                { label: 'Win Rate', value: `${stats.winRate.toFixed(1)}%`, good: stats.winRate >= 50 },
                { label: 'Profit Factor', value: stats.profitFactor.toFixed(2), good: stats.profitFactor >= 1.5 },
                { label: 'Avg Win', value: '₺3,250', good: true },
                { label: 'Avg Loss', value: '₺1,150', good: true },
                { label: 'Expectancy', value: '₺1,850', good: true },
                { label: 'Recovery Factor', value: '2.8', good: true },
              ].map((item) => (
                <div key={item.label} className="p-4 bg-surface-800 rounded-lg">
                  <p className="text-surface-400 text-sm">{item.label}</p>
                  <p className={`text-xl font-bold mt-1 ${item.good ? 'text-green-400' : 'text-yellow-400'}`}>{item.value}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
