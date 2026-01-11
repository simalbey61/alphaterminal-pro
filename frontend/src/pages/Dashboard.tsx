import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Zap, Brain, Wallet, Activity } from 'lucide-react';
import { formatCurrency, formatPercent } from '@/lib/utils';

const stats = [
  { title: 'Portföy Değeri', value: 1250000, change: 3.2, icon: Wallet },
  { title: 'Günlük P&L', value: 15420, change: 1.24, icon: TrendingUp },
  { title: 'Aktif Sinyaller', value: 12, icon: Zap },
  { title: 'AI Stratejiler', value: 8, icon: Brain },
];

const recentSignals = [
  { symbol: 'THYAO', direction: 'long', entry: 245.50, current: 252.30, pnl: 2.77 },
  { symbol: 'ASELS', direction: 'long', entry: 78.20, current: 81.45, pnl: 4.16 },
  { symbol: 'KCHOL', direction: 'short', entry: 142.00, current: 138.50, pnl: 2.46 },
  { symbol: 'EREGL', direction: 'long', entry: 52.80, current: 51.20, pnl: -3.03 },
];

export function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground">Portföy ve piyasa özeti</p>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <Card key={stat.title}>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">{stat.title}</CardTitle>
                <Icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {typeof stat.value === 'number' && stat.value > 1000 
                    ? formatCurrency(stat.value, 0) 
                    : stat.value}
                </div>
                {stat.change !== undefined && (
                  <p className={`text-xs ${stat.change >= 0 ? 'text-bull' : 'text-bear'}`}>
                    {stat.change >= 0 ? <TrendingUp className="inline h-3 w-3" /> : <TrendingDown className="inline h-3 w-3" />}
                    {' '}{formatPercent(stat.change)} bugün
                  </p>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Content Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Active Signals */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-primary" /> Aktif Sinyaller
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentSignals.map((signal) => (
                <div key={signal.symbol} className="flex items-center justify-between rounded-lg border p-3">
                  <div className="flex items-center gap-3">
                    <Badge variant={signal.direction === 'long' ? 'success' : 'danger'}>
                      {signal.direction.toUpperCase()}
                    </Badge>
                    <span className="font-semibold">{signal.symbol}</span>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm">{signal.current.toFixed(2)} ₺</div>
                    <div className={`text-xs ${signal.pnl >= 0 ? 'text-bull' : 'text-bear'}`}>
                      {formatPercent(signal.pnl)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Market Regime */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" /> Piyasa Rejimi
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Trend</span>
                <Badge variant="success">Bullish</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Volatilite</span>
                <Badge variant="warning">Normal</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Likidite</span>
                <Badge variant="secondary">Yüksek</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Faz</span>
                <Badge>Markup</Badge>
              </div>
              <div className="mt-4 rounded-lg bg-muted p-3">
                <p className="text-sm text-muted-foreground">
                  Piyasa bullish trend içinde, momentum stratejileri aktif.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
