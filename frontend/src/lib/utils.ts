import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import numeral from 'numeral';
import { format, formatDistanceToNow, parseISO } from 'date-fns';
import { tr } from 'date-fns/locale';

// ============================================================================
// CLASS NAME UTILITIES
// ============================================================================

/**
 * Merge Tailwind CSS classes with clsx
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// ============================================================================
// NUMBER FORMATTING
// ============================================================================

/**
 * Format number as currency (TRY)
 */
export function formatCurrency(value: number, decimals = 2): string {
  return numeral(value).format(`0,0.${'0'.repeat(decimals)}`) + ' ₺';
}

/**
 * Format number as USD
 */
export function formatUSD(value: number, decimals = 2): string {
  return '$' + numeral(value).format(`0,0.${'0'.repeat(decimals)}`);
}

/**
 * Format number with thousands separator
 */
export function formatNumber(value: number, decimals = 2): string {
  return numeral(value).format(`0,0.${'0'.repeat(decimals)}`);
}

/**
 * Format large numbers with abbreviations (1K, 1M, 1B)
 */
export function formatCompact(value: number): string {
  if (Math.abs(value) >= 1e9) {
    return numeral(value / 1e9).format('0.0') + 'B';
  }
  if (Math.abs(value) >= 1e6) {
    return numeral(value / 1e6).format('0.0') + 'M';
  }
  if (Math.abs(value) >= 1e3) {
    return numeral(value / 1e3).format('0.0') + 'K';
  }
  return numeral(value).format('0,0');
}

/**
 * Format percentage
 */
export function formatPercent(value: number, decimals = 2, showSign = true): string {
  const sign = showSign && value > 0 ? '+' : '';
  return sign + numeral(value).format(`0.${'0'.repeat(decimals)}`) + '%';
}

/**
 * Format price with appropriate decimals
 */
export function formatPrice(value: number): string {
  if (value >= 1000) {
    return numeral(value).format('0,0.00');
  }
  if (value >= 100) {
    return numeral(value).format('0,0.00');
  }
  if (value >= 1) {
    return numeral(value).format('0.00');
  }
  return numeral(value).format('0.0000');
}

/**
 * Format volume
 */
export function formatVolume(value: number): string {
  return formatCompact(value);
}

// ============================================================================
// DATE FORMATTING
// ============================================================================

/**
 * Format date
 */
export function formatDate(date: string | Date, pattern = 'dd MMM yyyy'): string {
  const d = typeof date === 'string' ? parseISO(date) : date;
  return format(d, pattern, { locale: tr });
}

/**
 * Format datetime
 */
export function formatDateTime(date: string | Date): string {
  const d = typeof date === 'string' ? parseISO(date) : date;
  return format(d, 'dd MMM yyyy HH:mm', { locale: tr });
}

/**
 * Format time
 */
export function formatTime(date: string | Date): string {
  const d = typeof date === 'string' ? parseISO(date) : date;
  return format(d, 'HH:mm:ss', { locale: tr });
}

/**
 * Format relative time (e.g., "5 dakika önce")
 */
export function formatRelativeTime(date: string | Date): string {
  const d = typeof date === 'string' ? parseISO(date) : date;
  return formatDistanceToNow(d, { addSuffix: true, locale: tr });
}

// ============================================================================
// TRADING UTILITIES
// ============================================================================

/**
 * Get color class based on value (positive/negative)
 */
export function getPnLColorClass(value: number): string {
  if (value > 0) {
    return 'text-bull';
  }
  if (value < 0) {
    return 'text-bear';
  }
  return 'text-muted-foreground';
}

/**
 * Get background color class based on value
 */
export function getPnLBgClass(value: number): string {
  if (value > 0) {
    return 'bg-bull-muted';
  }
  if (value < 0) {
    return 'bg-bear-muted';
  }
  return 'bg-muted';
}

/**
 * Format PnL with color
 */
export function formatPnL(value: number, isPercent = false): { text: string; colorClass: string } {
  const text = isPercent ? formatPercent(value) : formatCurrency(value);
  const colorClass = getPnLColorClass(value);
  return { text, colorClass };
}

/**
 * Calculate risk/reward ratio
 */
export function calculateRiskReward(
  entry: number,
  stopLoss: number,
  takeProfit: number
): number {
  const risk = Math.abs(entry - stopLoss);
  const reward = Math.abs(takeProfit - entry);
  return risk > 0 ? reward / risk : 0;
}

/**
 * Get regime color
 */
export function getRegimeColor(regime: string): string {
  const colors: Record<string, string> = {
    strong_bull: 'text-bull',
    bull: 'text-bull',
    weak_bull: 'text-bull/70',
    sideways: 'text-muted-foreground',
    weak_bear: 'text-bear/70',
    bear: 'text-bear',
    strong_bear: 'text-bear',
  };
  return colors[regime] ?? 'text-muted-foreground';
}

/**
 * Get lifecycle badge color
 */
export function getLifecycleColor(lifecycle: string): string {
  const colors: Record<string, string> = {
    discovered: 'bg-purple-500/20 text-purple-500',
    backtesting: 'bg-blue-500/20 text-blue-500',
    pending: 'bg-yellow-500/20 text-yellow-500',
    sandbox: 'bg-orange-500/20 text-orange-500',
    probation: 'bg-amber-500/20 text-amber-500',
    active: 'bg-green-500/20 text-green-500',
    paused: 'bg-gray-500/20 text-gray-500',
    retiring: 'bg-red-500/20 text-red-500',
    retired: 'bg-gray-600/20 text-gray-600',
  };
  return colors[lifecycle] ?? 'bg-muted text-muted-foreground';
}

/**
 * Get signal direction color
 */
export function getSignalDirectionColor(direction: 'long' | 'short'): string {
  return direction === 'long' ? 'text-bull' : 'text-bear';
}

// ============================================================================
// VALIDATION UTILITIES
// ============================================================================

/**
 * Check if email is valid
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Check if password is strong
 */
export function isStrongPassword(password: string): {
  isValid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (password.length < 8) {
    errors.push('En az 8 karakter olmalı');
  }
  if (!/[A-Z]/.test(password)) {
    errors.push('En az 1 büyük harf içermeli');
  }
  if (!/[a-z]/.test(password)) {
    errors.push('En az 1 küçük harf içermeli');
  }
  if (!/[0-9]/.test(password)) {
    errors.push('En az 1 rakam içermeli');
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
}

// ============================================================================
// MISC UTILITIES
// ============================================================================

/**
 * Sleep for specified milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Generate unique ID
 */
export function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

/**
 * Debounce function
 */
export function debounce<T extends (...args: Parameters<T>) => ReturnType<T>>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

/**
 * Throttle function
 */
export function throttle<T extends (...args: Parameters<T>) => ReturnType<T>>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      fn(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

/**
 * Copy text to clipboard
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/**
 * Download data as JSON file
 */
export function downloadJSON(data: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
