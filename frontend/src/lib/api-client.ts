import axios, {
  type AxiosInstance,
  type AxiosError,
  type InternalAxiosRequestConfig,
  type AxiosResponse,
} from 'axios';
import { useAuthStore } from '@/stores/auth-store';

// ============================================================================
// CONFIGURATION
// ============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';
const REQUEST_TIMEOUT = 30000; // 30 seconds

// ============================================================================
// AXIOS INSTANCE
// ============================================================================

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: REQUEST_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

// ============================================================================
// REQUEST INTERCEPTOR
// ============================================================================

apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Get token from store
    const token = useAuthStore.getState().tokens?.accessToken;

    // Add auth header if token exists
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    // Add request ID for tracking
    config.headers['X-Request-ID'] = crypto.randomUUID();

    return config;
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

// ============================================================================
// RESPONSE INTERCEPTOR
// ============================================================================

apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & {
      _retry?: boolean;
    };

    // Handle 401 Unauthorized
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        // Attempt to refresh token
        const refreshToken = useAuthStore.getState().tokens?.refreshToken;

        if (refreshToken) {
          const response = await axios.post<{
            accessToken: string;
            refreshToken: string;
            expiresAt: number;
          }>(`${API_BASE_URL}/auth/refresh`, {
            refreshToken,
          });

          // Update tokens in store
          useAuthStore.getState().setTokens(response.data);

          // Retry original request
          originalRequest.headers.Authorization = `Bearer ${response.data.accessToken}`;
          return apiClient(originalRequest);
        }
      } catch {
        // Refresh failed, logout user
        useAuthStore.getState().logout();
        window.location.href = '/login';
      }
    }

    // Handle other errors
    return Promise.reject(formatApiError(error));
  }
);

// ============================================================================
// ERROR HANDLING
// ============================================================================

export interface FormattedApiError {
  message: string;
  code: string;
  status: number;
  details?: Record<string, unknown>;
}

function formatApiError(error: AxiosError): FormattedApiError {
  if (error.response) {
    // Server responded with error
    const data = error.response.data as Record<string, unknown>;

    return {
      message: (data.message as string) || (data.detail as string) || 'Bir hata oluştu',
      code: (data.code as string) || 'SERVER_ERROR',
      status: error.response.status,
      details: data,
    };
  }

  if (error.request) {
    // Request made but no response
    return {
      message: 'Sunucuya ulaşılamıyor',
      code: 'NETWORK_ERROR',
      status: 0,
    };
  }

  // Request setup error
  return {
    message: error.message || 'Bir hata oluştu',
    code: 'REQUEST_ERROR',
    status: 0,
  };
}

// ============================================================================
// API HELPER FUNCTIONS
// ============================================================================

/**
 * GET request
 */
export async function get<T>(url: string, params?: Record<string, unknown>): Promise<T> {
  const response = await apiClient.get<T>(url, { params });
  return response.data;
}

/**
 * POST request
 */
export async function post<T>(url: string, data?: unknown): Promise<T> {
  const response = await apiClient.post<T>(url, data);
  return response.data;
}

/**
 * PUT request
 */
export async function put<T>(url: string, data?: unknown): Promise<T> {
  const response = await apiClient.put<T>(url, data);
  return response.data;
}

/**
 * PATCH request
 */
export async function patch<T>(url: string, data?: unknown): Promise<T> {
  const response = await apiClient.patch<T>(url, data);
  return response.data;
}

/**
 * DELETE request
 */
export async function del<T>(url: string): Promise<T> {
  const response = await apiClient.delete<T>(url);
  return response.data;
}

// ============================================================================
// API ENDPOINTS
// ============================================================================

export const api = {
  // Auth
  auth: {
    login: (data: { email: string; password: string }) =>
      post<{ user: unknown; tokens: unknown }>('/auth/login', data),
    register: (data: { email: string; password: string; username: string }) =>
      post<{ user: unknown; tokens: unknown }>('/auth/register', data),
    logout: () => post<void>('/auth/logout'),
    refresh: (refreshToken: string) =>
      post<{ accessToken: string; refreshToken: string; expiresAt: number }>(
        '/auth/refresh',
        { refreshToken }
      ),
    me: () => get<unknown>('/auth/me'),
  },

  // Stocks
  stocks: {
    list: (params?: { search?: string; sector?: string; page?: number; pageSize?: number }) =>
      get<unknown>('/stocks', params),
    get: (symbol: string) => get<unknown>(`/stocks/${symbol}`),
    quote: (symbol: string) => get<unknown>(`/stocks/${symbol}/quote`),
    ohlcv: (symbol: string, timeframe: string, limit?: number) =>
      get<unknown>(`/stocks/${symbol}/ohlcv`, { timeframe, limit }),
    search: (query: string) => get<unknown>('/stocks/search', { query }),
  },

  // Signals
  signals: {
    list: (params?: { status?: string; symbol?: string; page?: number }) =>
      get<unknown>('/signals', params),
    get: (id: string) => get<unknown>(`/signals/${id}`),
    active: () => get<unknown>('/signals/active'),
    history: (params?: { page?: number; pageSize?: number }) =>
      get<unknown>('/signals/history', params),
  },

  // Strategies
  strategies: {
    list: (params?: { lifecycle?: string; type?: string }) =>
      get<unknown>('/strategies', params),
    get: (id: string) => get<unknown>(`/strategies/${id}`),
    performance: (id: string) => get<unknown>(`/strategies/${id}/performance`),
  },

  // AI Strategy
  aiStrategy: {
    health: () => get<unknown>('/ai-strategy/health'),
    discover: (data: { symbol: string; timeframe: string; lookbackDays: number }) =>
      post<unknown>('/ai-strategy/discover', data),
    signals: (data: { symbol: string; timeframe: string; entryPrice?: number }) =>
      post<unknown>('/ai-strategy/signals', data),
    regime: (symbol: string, timeframe: string) =>
      get<unknown>('/ai-strategy/regime', { symbol, timeframe }),
    diversity: () => get<unknown>('/ai-strategy/diversity'),
    evolution: () => post<unknown>('/ai-strategy/evolution/run'),
  },

  // Analysis
  analysis: {
    full: (symbol: string, timeframe: string) =>
      get<unknown>(`/analysis/${symbol}`, { timeframe }),
    smc: (symbol: string, timeframe: string) =>
      get<unknown>(`/analysis/${symbol}/smc`, { timeframe }),
    orderflow: (symbol: string, timeframe: string) =>
      get<unknown>(`/analysis/${symbol}/orderflow`, { timeframe }),
    alpha: (symbol: string) => get<unknown>(`/analysis/${symbol}/alpha`),
  },

  // Portfolio
  portfolio: {
    get: () => get<unknown>('/portfolio'),
    positions: () => get<unknown>('/portfolio/positions'),
    trades: (params?: { page?: number; pageSize?: number }) =>
      get<unknown>('/portfolio/trades', params),
    summary: () => get<unknown>('/portfolio/summary'),
  },

  // Market
  market: {
    overview: () => get<unknown>('/market/overview'),
    sectors: () => get<unknown>('/market/sectors'),
    gainers: () => get<unknown>('/market/gainers'),
    losers: () => get<unknown>('/market/losers'),
    mostActive: () => get<unknown>('/market/most-active'),
  },

  // User
  user: {
    profile: () => get<unknown>('/users/me'),
    updateProfile: (data: unknown) => patch<unknown>('/users/me', data),
    preferences: () => get<unknown>('/users/me/preferences'),
    updatePreferences: (data: unknown) => patch<unknown>('/users/me/preferences', data),
  },
};

export default apiClient;
