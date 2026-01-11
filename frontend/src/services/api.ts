/**
 * AlphaTerminal Pro - API Service
 * ================================
 * 
 * HTTP client for backend API communication.
 */

import type {
  APIResponse,
  OHLCVBar,
  TradingSignal,
  BacktestConfig,
  BacktestResult,
  PortfolioSummary,
  DiscoveredStrategy,
  TimeFrame,
} from '../types';

// =============================================================================
// CONFIG
// =============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v2';
const API_TIMEOUT = 30000;

// =============================================================================
// HTTP CLIENT
// =============================================================================

class APIClient {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl: string = API_BASE_URL, timeout: number = API_TIMEOUT) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<APIResponse<T>> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      const data = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: {
            code: data.error?.code || 'UNKNOWN_ERROR',
            message: data.error?.message || 'An error occurred',
            details: data.error?.details,
          },
        };
      }

      return {
        success: true,
        data: data.data || data,
        meta: data.meta,
      };
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error && error.name === 'AbortError') {
        return {
          success: false,
          error: {
            code: 'TIMEOUT',
            message: 'Request timed out',
          },
        };
      }

      return {
        success: false,
        error: {
          code: 'NETWORK_ERROR',
          message: error instanceof Error ? error.message : 'Network error',
        },
      };
    }
  }

  async get<T>(endpoint: string): Promise<APIResponse<T>> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, body: unknown): Promise<APIResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  async put<T>(endpoint: string, body: unknown): Promise<APIResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: JSON.stringify(body),
    });
  }

  async delete<T>(endpoint: string): Promise<APIResponse<T>> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }
}

// =============================================================================
// API INSTANCE
// =============================================================================

const api = new APIClient();

// =============================================================================
// MARKET DATA API
// =============================================================================

export const marketAPI = {
  async getOHLCV(
    symbol: string,
    interval: TimeFrame,
    limit: number = 500
  ): Promise<APIResponse<OHLCVBar[]>> {
    return api.get<OHLCVBar[]>(
      `/market/ohlcv/${symbol}?interval=${interval}&limit=${limit}`
    );
  },

  async getTicker(symbol: string): Promise<APIResponse<{ price: number; change: number }>> {
    return api.get(`/market/ticker/${symbol}`);
  },

  async getSymbols(): Promise<APIResponse<string[]>> {
    return api.get<string[]>('/market/symbols');
  },

  async searchSymbols(query: string): Promise<APIResponse<string[]>> {
    return api.get<string[]>(`/market/search?q=${encodeURIComponent(query)}`);
  },
};

// =============================================================================
// SIGNALS API
// =============================================================================

export const signalsAPI = {
  async getActiveSignals(): Promise<APIResponse<TradingSignal[]>> {
    return api.get<TradingSignal[]>('/signals/active');
  },

  async getSignalsBySymbol(symbol: string): Promise<APIResponse<TradingSignal[]>> {
    return api.get<TradingSignal[]>(`/signals/symbol/${symbol}`);
  },

  async generateSignals(
    symbol: string,
    interval: TimeFrame
  ): Promise<APIResponse<TradingSignal[]>> {
    return api.post<TradingSignal[]>('/signals/generate', { symbol, interval });
  },
};

// =============================================================================
// BACKTEST API
// =============================================================================

export const backtestAPI = {
  async runBacktest(config: BacktestConfig): Promise<APIResponse<BacktestResult>> {
    return api.post<BacktestResult>('/backtest/run', config);
  },

  async getBacktest(id: string): Promise<APIResponse<BacktestResult>> {
    return api.get<BacktestResult>(`/backtest/${id}`);
  },

  async getBacktests(): Promise<APIResponse<BacktestResult[]>> {
    return api.get<BacktestResult[]>('/backtest/list');
  },

  async getStrategies(): Promise<APIResponse<{ name: string; params: Record<string, unknown> }[]>> {
    return api.get('/backtest/strategies');
  },
};

// =============================================================================
// PORTFOLIO API
// =============================================================================

export const portfolioAPI = {
  async getSummary(): Promise<APIResponse<PortfolioSummary>> {
    return api.get<PortfolioSummary>('/portfolio/summary');
  },

  async getHistory(days: number = 30): Promise<APIResponse<{ date: string; value: number }[]>> {
    return api.get(`/portfolio/history?days=${days}`);
  },
};

// =============================================================================
// ML API
// =============================================================================

export const mlAPI = {
  async discoverStrategies(
    symbol: string,
    interval: TimeFrame
  ): Promise<APIResponse<DiscoveredStrategy[]>> {
    return api.post<DiscoveredStrategy[]>('/ml/discover', { symbol, interval });
  },

  async getDiscoveredStrategies(): Promise<APIResponse<DiscoveredStrategy[]>> {
    return api.get<DiscoveredStrategy[]>('/ml/strategies');
  },

  async trainModel(
    symbol: string,
    interval: TimeFrame,
    modelType: string
  ): Promise<APIResponse<{ modelId: string; accuracy: number }>> {
    return api.post('/ml/train', { symbol, interval, modelType });
  },
};

// =============================================================================
// HEALTH API
// =============================================================================

export const healthAPI = {
  async check(): Promise<APIResponse<{ status: string; version: string }>> {
    return api.get('/health');
  },
};

export { api, APIClient };
export default api;
