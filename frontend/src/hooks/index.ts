/**
 * AlphaTerminal Pro - Custom React Hooks
 * =======================================
 * 
 * Reusable hooks for data fetching and state management.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  OHLCVBar,
  TradingSignal,
  BacktestResult,
  PortfolioSummary,
  TimeFrame,
} from '../types';
import { marketAPI, signalsAPI, backtestAPI, portfolioAPI } from '../services/api';

// =============================================================================
// useAsync - Generic async data fetching hook
// =============================================================================

interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function useAsync<T>(
  asyncFunction: () => Promise<{ success: boolean; data?: T; error?: { message: string } }>,
  dependencies: unknown[] = []
): AsyncState<T> & { refetch: () => void } {
  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    loading: true,
    error: null,
  });

  const execute = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const result = await asyncFunction();
      
      if (result.success && result.data) {
        setState({ data: result.data, loading: false, error: null });
      } else {
        setState({
          data: null,
          loading: false,
          error: result.error?.message || 'An error occurred',
        });
      }
    } catch (err) {
      setState({
        data: null,
        loading: false,
        error: err instanceof Error ? err.message : 'An error occurred',
      });
    }
  }, dependencies);

  useEffect(() => {
    execute();
  }, [execute]);

  return { ...state, refetch: execute };
}

// =============================================================================
// useOHLCV - Fetch OHLCV data
// =============================================================================

export function useOHLCV(
  symbol: string | null,
  interval: TimeFrame,
  limit: number = 500
) {
  return useAsync<OHLCVBar[]>(
    () => symbol ? marketAPI.getOHLCV(symbol, interval, limit) : Promise.resolve({ success: false }),
    [symbol, interval, limit]
  );
}

// =============================================================================
// useSignals - Fetch trading signals
// =============================================================================

export function useSignals(symbol?: string) {
  return useAsync<TradingSignal[]>(
    () => symbol 
      ? signalsAPI.getSignalsBySymbol(symbol)
      : signalsAPI.getActiveSignals(),
    [symbol]
  );
}

// =============================================================================
// useBacktests - Fetch backtest results
// =============================================================================

export function useBacktests() {
  return useAsync<BacktestResult[]>(
    () => backtestAPI.getBacktests(),
    []
  );
}

// =============================================================================
// usePortfolio - Fetch portfolio summary
// =============================================================================

export function usePortfolio() {
  return useAsync<PortfolioSummary>(
    () => portfolioAPI.getSummary(),
    []
  );
}

// =============================================================================
// useSymbolSearch - Search symbols with debouncing
// =============================================================================

export function useSymbolSearch(query: string, debounceMs: number = 300) {
  const [results, setResults] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    if (!query || query.length < 2) {
      setResults([]);
      return;
    }

    setLoading(true);

    timeoutRef.current = setTimeout(async () => {
      const response = await marketAPI.searchSymbols(query);
      
      if (response.success && response.data) {
        setResults(response.data);
      }
      
      setLoading(false);
    }, debounceMs);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [query, debounceMs]);

  return { results, loading };
}

// =============================================================================
// useWebSocket - WebSocket connection for real-time data
// =============================================================================

interface WebSocketOptions {
  url: string;
  onMessage?: (data: unknown) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
}

export function useWebSocket(options: WebSocketOptions) {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<unknown>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    try {
      wsRef.current = new WebSocket(options.url);

      wsRef.current.onopen = () => {
        setConnected(true);
        options.onOpen?.();
      };

      wsRef.current.onclose = () => {
        setConnected(false);
        options.onClose?.();

        if (options.reconnect) {
          reconnectTimeoutRef.current = setTimeout(
            connect,
            options.reconnectInterval || 5000
          );
        }
      };

      wsRef.current.onerror = (error) => {
        options.onError?.(error);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
          options.onMessage?.(data);
        } catch {
          setLastMessage(event.data);
          options.onMessage?.(event.data);
        }
      };
    } catch (error) {
      console.error('WebSocket connection error:', error);
    }
  }, [options]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    wsRef.current?.close();
  }, []);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return { connected, lastMessage, send, disconnect, reconnect: connect };
}

// =============================================================================
// useLocalStorage - Persist state in localStorage
// =============================================================================

export function useLocalStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((prev: T) => T)) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = useCallback(
    (value: T | ((prev: T) => T)) => {
      try {
        const valueToStore = value instanceof Function ? value(storedValue) : value;
        setStoredValue(valueToStore);
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
      } catch (error) {
        console.error('Error saving to localStorage:', error);
      }
    },
    [key, storedValue]
  );

  return [storedValue, setValue];
}

// =============================================================================
// useInterval - Run callback at interval
// =============================================================================

export function useInterval(callback: () => void, delay: number | null) {
  const savedCallback = useRef(callback);

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay === null) return;

    const tick = () => savedCallback.current();
    const id = setInterval(tick, delay);

    return () => clearInterval(id);
  }, [delay]);
}

// =============================================================================
// useTheme - Theme management
// =============================================================================

export function useTheme() {
  const [theme, setTheme] = useLocalStorage<'light' | 'dark'>('theme', 'dark');

  const toggleTheme = useCallback(() => {
    setTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  }, [setTheme]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  return { theme, setTheme, toggleTheme };
}
