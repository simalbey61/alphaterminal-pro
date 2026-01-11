import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User, AuthTokens } from '@/types';

interface AuthState {
  user: User | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (user: User, tokens: AuthTokens) => void;
  logout: () => void;
  setLoading: (loading: boolean) => void;
  setTokens: (tokens: AuthTokens) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      tokens: null,
      isAuthenticated: false,
      isLoading: false,
      login: (user, tokens) => set({ user, tokens, isAuthenticated: true, isLoading: false }),
      logout: () => set({ user: null, tokens: null, isAuthenticated: false }),
      setLoading: (isLoading) => set({ isLoading }),
      setTokens: (tokens) => set({ tokens }),
    }),
    { name: 'alpha-terminal-auth', partialize: (s) => ({ user: s.user, tokens: s.tokens, isAuthenticated: s.isAuthenticated }) }
  )
);

export const useAuth = () => {
  const store = useAuthStore();
  return store;
};
