/**
 * AlphaTerminal Pro - TypeScript Type Definitions
 * ================================================
 * 
 * Core types for the trading terminal frontend.
 */

// =============================================================================
// ENUMS
// =============================================================================

export enum SignalType {
  BUY = 'buy',
  SELL = 'sell',
  HOLD = 'hold',
  CLOSE = 'close',
}

export enum TimeFrame {
  M1 = '1m',
  M5 = '5m',
  M15 = '15m',
  M30 = '30m',
  H1 = '1h',
  H4 = '4h',
  D1 = '1d',
  W1 = '1w',
}

export enum OrderType {
  MARKET = 'market',
  LIMIT = 'limit',
  STOP = 'stop',
  STOP_LIMIT = 'stop_limit',
}

export enum PositionSide {
  LONG = 'long',
  SHORT = 'short',
}

export enum BacktestStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

// =============================================================================
// MARKET DATA
// =============================================================================

export interface OHLCVBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Ticker {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high24h: number;
  low24h: number;
  lastUpdate: string;
}

export interface MarketDepth {
  bids: [number, number][]; // [price, size]
  asks: [number, number][];
  timestamp: string;
}

// =============================================================================
// TRADING SIGNALS
// =============================================================================

export interface TradingSignal {
  id: string;
  symbol: string;
  signalType: SignalType;
  price: number;
  targetPrice?: number;
  stopLoss?: number;
  confidence: number;
  strategy: string;
  timeframe: TimeFrame;
  createdAt: string;
  expiresAt?: string;
  indicators?: Record<string, number>;
  notes?: string;
}

export interface SignalSummary {
  totalSignals: number;
  buySignals: number;
  sellSignals: number;
  avgConfidence: number;
  successRate: number;
}

// =============================================================================
// BACKTEST
// =============================================================================

export interface BacktestConfig {
  symbol: string;
  interval: TimeFrame;
  startDate: string;
  endDate: string;
  initialCapital: number;
  commission: number;
  slippage: number;
  strategy: StrategyConfig;
}

export interface StrategyConfig {
  name: string;
  type: string;
  params: Record<string, number | string | boolean>;
}

export interface BacktestResult {
  id: string;
  status: BacktestStatus;
  config: BacktestConfig;
  
  // Performance
  totalReturn: number;
  totalReturnPct: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  volatility: number;
  
  // Trade stats
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  profitFactor: number;
  avgTrade: number;
  avgWinner: number;
  avgLoser: number;
  largestWinner: number;
  largestLoser: number;
  avgHoldingHours: number;
  
  // Data
  equityCurve?: number[];
  trades?: Trade[];
  monthlyReturns?: Record<string, number>;
  
  createdAt: string;
  completedAt?: string;
}

export interface Trade {
  id: string;
  entryTime: string;
  exitTime: string;
  side: PositionSide;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  pnl: number;
  pnlPercent: number;
  holdingHours: number;
  exitReason: string;
}

// =============================================================================
// PORTFOLIO
// =============================================================================

export interface Position {
  symbol: string;
  side: PositionSide;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  marketValue: number;
  openedAt: string;
}

export interface PortfolioSummary {
  totalValue: number;
  cashBalance: number;
  positionsValue: number;
  unrealizedPnl: number;
  realizedPnl: number;
  dailyPnl: number;
  dailyPnlPercent: number;
  positions: Position[];
}

// =============================================================================
// ML / STRATEGY DISCOVERY
// =============================================================================

export interface MLModel {
  id: string;
  name: string;
  type: 'decision_tree' | 'random_forest' | 'gradient_boosting' | 'mlp' | 'lstm';
  symbol: string;
  interval: TimeFrame;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  trainedAt: string;
  features: string[];
}

export interface DiscoveredStrategy {
  id: string;
  name: string;
  symbol: string;
  interval: TimeFrame;
  
  // Performance metrics
  expectedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  
  // Model info
  modelType: string;
  confidence: number;
  
  // Rules
  entryConditions: string[];
  exitConditions: string[];
  
  createdAt: string;
}

// =============================================================================
// API
// =============================================================================

export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: APIError;
  meta?: APIMeta;
}

export interface APIError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface APIMeta {
  requestId: string;
  timestamp: string;
  processingTimeMs: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

// =============================================================================
// UI STATE
// =============================================================================

export interface AppState {
  theme: 'light' | 'dark';
  sidebarCollapsed: boolean;
  selectedSymbol: string | null;
  selectedTimeframe: TimeFrame;
  notifications: Notification[];
}

export interface AppNotification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
}

// =============================================================================
// CHART
// =============================================================================

export interface ChartOptions {
  showVolume: boolean;
  showIndicators: boolean;
  indicators: IndicatorConfig[];
  annotations: ChartAnnotation[];
}

export interface IndicatorConfig {
  type: string;
  params: Record<string, number>;
  color: string;
  visible: boolean;
}

export interface ChartAnnotation {
  type: 'line' | 'rect' | 'text' | 'signal';
  coordinates: { x: number | string; y: number }[];
  style?: Record<string, string | number>;
  label?: string;
}
