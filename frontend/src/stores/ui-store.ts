import { create } from 'zustand';
import { persist } from 'zustand/middleware';

type Theme = 'light' | 'dark' | 'system';

interface UIState {
  theme: Theme;
  sidebarCollapsed: boolean;
  setTheme: (theme: Theme) => void;
  toggleSidebar: () => void;
}

const applyTheme = (theme: Theme) => {
  const root = document.documentElement;
  root.classList.remove('light', 'dark');
  const resolved = theme === 'system' 
    ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
    : theme;
  root.classList.add(resolved);
};

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      theme: 'dark',
      sidebarCollapsed: false,
      setTheme: (theme) => { applyTheme(theme); set({ theme }); },
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
    }),
    { name: 'alpha-terminal-ui' }
  )
);

if (typeof window !== 'undefined') {
  const stored = localStorage.getItem('alpha-terminal-ui');
  if (stored) {
    try { applyTheme(JSON.parse(stored).state?.theme || 'dark'); } catch { applyTheme('dark'); }
  }
}
