/**
 * AlphaTerminal Pro - Chart Components
 * =====================================
 * 
 * Trading chart components using lightweight-charts.
 */

import React, { useEffect, useRef, useState } from 'react';
import type { OHLCVBar, TradingSignal, SignalType } from '../../types';

// =============================================================================
// TYPES
// =============================================================================

interface CandlestickChartProps {
  data: OHLCVBar[];
  signals?: TradingSignal[];
  height?: number;
  showVolume?: boolean;
  theme?: 'light' | 'dark';
  onCrosshairMove?: (price: number | null, time: string | null) => void;
}

interface EquityCurveChartProps {
  data: number[];
  benchmark?: number[];
  height?: number;
  theme?: 'light' | 'dark';
}

interface DrawdownChartProps {
  data: number[];
  height?: number;
  theme?: 'light' | 'dark';
}

// =============================================================================
// CANDLESTICK CHART (SVG Implementation)
// =============================================================================

export const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  signals = [],
  height = 400,
  showVolume = true,
  theme = 'dark',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height });

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height,
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [height]);

  if (!data || data.length === 0) {
    return (
      <div className="chart-empty">
        <p>No data available</p>
      </div>
    );
  }

  // Calculate scales
  const padding = { top: 20, right: 60, bottom: showVolume ? 80 : 40, left: 10 };
  const chartHeight = showVolume ? dimensions.height * 0.7 : dimensions.height - padding.top - padding.bottom;
  const volumeHeight = showVolume ? dimensions.height * 0.2 : 0;

  const prices = data.flatMap(d => [d.high, d.low]);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const priceRange = maxPrice - minPrice || 1;

  const maxVolume = Math.max(...data.map(d => d.volume));

  const candleWidth = Math.max(2, (dimensions.width - padding.left - padding.right) / data.length - 1);

  const scaleY = (price: number) => {
    return padding.top + chartHeight - ((price - minPrice) / priceRange) * chartHeight;
  };

  const scaleVolumeY = (volume: number) => {
    const baseY = padding.top + chartHeight + 10;
    return baseY + volumeHeight - (volume / maxVolume) * volumeHeight;
  };

  // Colors
  const colors = theme === 'dark' ? {
    bg: '#1E1E1E',
    text: '#888888',
    grid: '#333333',
    bullish: '#26A69A',
    bearish: '#EF5350',
    volume: '#455A64',
  } : {
    bg: '#FFFFFF',
    text: '#666666',
    grid: '#EEEEEE',
    bullish: '#26A69A',
    bearish: '#EF5350',
    volume: '#90A4AE',
  };

  // Generate price labels
  const priceLabels = [];
  const labelCount = 5;
  for (let i = 0; i <= labelCount; i++) {
    const price = minPrice + (priceRange * i / labelCount);
    priceLabels.push({
      price,
      y: scaleY(price),
    });
  }

  return (
    <div ref={containerRef} className="chart-container">
      <svg
        width={dimensions.width}
        height={dimensions.height}
        style={{ background: colors.bg }}
      >
        {/* Grid lines */}
        {priceLabels.map((label, i) => (
          <g key={i}>
            <line
              x1={padding.left}
              y1={label.y}
              x2={dimensions.width - padding.right}
              y2={label.y}
              stroke={colors.grid}
              strokeDasharray="2,2"
            />
            <text
              x={dimensions.width - padding.right + 5}
              y={label.y + 4}
              fill={colors.text}
              fontSize="10"
            >
              {label.price.toFixed(2)}
            </text>
          </g>
        ))}

        {/* Candlesticks */}
        {data.map((bar, i) => {
          const x = padding.left + i * (candleWidth + 1);
          const isBullish = bar.close >= bar.open;
          const color = isBullish ? colors.bullish : colors.bearish;

          const bodyTop = scaleY(Math.max(bar.open, bar.close));
          const bodyBottom = scaleY(Math.min(bar.open, bar.close));
          const bodyHeight = Math.max(1, bodyBottom - bodyTop);

          return (
            <g key={i}>
              {/* Wick */}
              <line
                x1={x + candleWidth / 2}
                y1={scaleY(bar.high)}
                x2={x + candleWidth / 2}
                y2={scaleY(bar.low)}
                stroke={color}
                strokeWidth="1"
              />
              {/* Body */}
              <rect
                x={x}
                y={bodyTop}
                width={candleWidth}
                height={bodyHeight}
                fill={isBullish ? 'transparent' : color}
                stroke={color}
                strokeWidth="1"
              />
            </g>
          );
        })}

        {/* Volume bars */}
        {showVolume && data.map((bar, i) => {
          const x = padding.left + i * (candleWidth + 1);
          const isBullish = bar.close >= bar.open;
          const volY = scaleVolumeY(bar.volume);
          const volHeight = padding.top + chartHeight + 10 + volumeHeight - volY;

          return (
            <rect
              key={`vol-${i}`}
              x={x}
              y={volY}
              width={candleWidth}
              height={volHeight}
              fill={isBullish ? colors.bullish : colors.bearish}
              opacity="0.5"
            />
          );
        })}

        {/* Signal markers */}
        {signals.map((signal, i) => {
          // Find matching bar by timestamp
          const barIndex = data.findIndex(d => 
            d.timestamp.includes(signal.createdAt.split('T')[0])
          );
          
          if (barIndex === -1) return null;

          const x = padding.left + barIndex * (candleWidth + 1) + candleWidth / 2;
          const bar = data[barIndex];
          const isBuy = signal.signalType === 'buy' as SignalType;
          const y = isBuy ? scaleY(bar.low) + 10 : scaleY(bar.high) - 10;
          const color = isBuy ? colors.bullish : colors.bearish;

          return (
            <g key={`signal-${i}`}>
              <polygon
                points={isBuy
                  ? `${x},${y} ${x-6},${y+10} ${x+6},${y+10}`
                  : `${x},${y} ${x-6},${y-10} ${x+6},${y-10}`
                }
                fill={color}
              />
            </g>
          );
        })}
      </svg>
    </div>
  );
};

// =============================================================================
// EQUITY CURVE CHART
// =============================================================================

export const EquityCurveChart: React.FC<EquityCurveChartProps> = ({
  data,
  benchmark,
  height = 300,
  theme = 'dark',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);

  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setWidth(containerRef.current.offsetWidth);
      }
    };

    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  if (!data || data.length === 0) {
    return <div className="chart-empty"><p>No data</p></div>;
  }

  const padding = { top: 20, right: 60, bottom: 30, left: 10 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const allValues = benchmark ? [...data, ...benchmark] : data;
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const range = maxValue - minValue || 1;

  const scaleX = (index: number) => padding.left + (index / (data.length - 1)) * chartWidth;
  const scaleY = (value: number) => padding.top + chartHeight - ((value - minValue) / range) * chartHeight;

  const colors = theme === 'dark' ? {
    bg: '#1E1E1E',
    primary: '#2962FF',
    benchmark: '#78909C',
    grid: '#333333',
    text: '#888888',
    positive: '#4CAF50',
    negative: '#F44336',
  } : {
    bg: '#FFFFFF',
    primary: '#2962FF',
    benchmark: '#78909C',
    grid: '#EEEEEE',
    text: '#666666',
    positive: '#4CAF50',
    negative: '#F44336',
  };

  // Create path
  const createPath = (values: number[]) => {
    return values.map((v, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i)} ${scaleY(v)}`).join(' ');
  };

  const initialValue = data[0];
  const finalValue = data[data.length - 1];
  const isPositive = finalValue >= initialValue;

  return (
    <div ref={containerRef} className="chart-container">
      <svg width={width} height={height} style={{ background: colors.bg }}>
        {/* Grid */}
        {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
          const y = padding.top + chartHeight * (1 - pct);
          const value = minValue + range * pct;
          return (
            <g key={i}>
              <line
                x1={padding.left}
                y1={y}
                x2={width - padding.right}
                y2={y}
                stroke={colors.grid}
                strokeDasharray="2,2"
              />
              <text x={width - padding.right + 5} y={y + 4} fill={colors.text} fontSize="10">
                {value.toFixed(0)}
              </text>
            </g>
          );
        })}

        {/* Initial value line */}
        <line
          x1={padding.left}
          y1={scaleY(initialValue)}
          x2={width - padding.right}
          y2={scaleY(initialValue)}
          stroke={colors.text}
          strokeDasharray="4,4"
          opacity="0.5"
        />

        {/* Fill area */}
        <path
          d={`${createPath(data)} L ${scaleX(data.length - 1)} ${scaleY(initialValue)} L ${scaleX(0)} ${scaleY(initialValue)} Z`}
          fill={isPositive ? colors.positive : colors.negative}
          opacity="0.1"
        />

        {/* Benchmark line */}
        {benchmark && (
          <path
            d={createPath(benchmark)}
            fill="none"
            stroke={colors.benchmark}
            strokeWidth="1.5"
            strokeDasharray="4,4"
          />
        )}

        {/* Main line */}
        <path
          d={createPath(data)}
          fill="none"
          stroke={colors.primary}
          strokeWidth="2"
        />

        {/* End point */}
        <circle
          cx={scaleX(data.length - 1)}
          cy={scaleY(finalValue)}
          r="4"
          fill={isPositive ? colors.positive : colors.negative}
        />
      </svg>
    </div>
  );
};

// =============================================================================
// DRAWDOWN CHART
// =============================================================================

export const DrawdownChart: React.FC<DrawdownChartProps> = ({
  data,
  height = 200,
  theme = 'dark',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);

  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setWidth(containerRef.current.offsetWidth);
      }
    };

    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  if (!data || data.length === 0) {
    return <div className="chart-empty"><p>No data</p></div>;
  }

  // Calculate drawdown from equity
  const runningMax = data.reduce<number[]>((acc, val, i) => {
    acc.push(i === 0 ? val : Math.max(acc[i - 1], val));
    return acc;
  }, []);

  const drawdown = data.map((val, i) => ((val - runningMax[i]) / runningMax[i]) * 100);

  const padding = { top: 10, right: 60, bottom: 20, left: 10 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const minDD = Math.min(...drawdown, -1);

  const scaleX = (index: number) => padding.left + (index / (drawdown.length - 1)) * chartWidth;
  const scaleY = (value: number) => padding.top + (-value / minDD) * chartHeight;

  const colors = theme === 'dark' ? {
    bg: '#1E1E1E',
    fill: '#EF5350',
    stroke: '#F44336',
    grid: '#333333',
    text: '#888888',
  } : {
    bg: '#FFFFFF',
    fill: '#FFCDD2',
    stroke: '#F44336',
    grid: '#EEEEEE',
    text: '#666666',
  };

  const pathData = drawdown.map((v, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i)} ${scaleY(v)}`).join(' ');
  const fillPath = `${pathData} L ${scaleX(drawdown.length - 1)} ${padding.top} L ${scaleX(0)} ${padding.top} Z`;

  const maxDDIndex = drawdown.indexOf(Math.min(...drawdown));
  const maxDD = drawdown[maxDDIndex];

  return (
    <div ref={containerRef} className="chart-container">
      <svg width={width} height={height} style={{ background: colors.bg }}>
        {/* Zero line */}
        <line
          x1={padding.left}
          y1={padding.top}
          x2={width - padding.right}
          y2={padding.top}
          stroke={colors.grid}
        />

        {/* Fill */}
        <path d={fillPath} fill={colors.fill} opacity="0.3" />

        {/* Line */}
        <path d={pathData} fill="none" stroke={colors.stroke} strokeWidth="1.5" />

        {/* Max drawdown marker */}
        <circle
          cx={scaleX(maxDDIndex)}
          cy={scaleY(maxDD)}
          r="4"
          fill={colors.stroke}
        />
        <text
          x={scaleX(maxDDIndex) + 8}
          y={scaleY(maxDD) + 4}
          fill={colors.text}
          fontSize="10"
        >
          {maxDD.toFixed(1)}%
        </text>

        {/* Y-axis labels */}
        <text x={width - padding.right + 5} y={padding.top + 4} fill={colors.text} fontSize="10">0%</text>
        <text x={width - padding.right + 5} y={height - padding.bottom} fill={colors.text} fontSize="10">
          {minDD.toFixed(0)}%
        </text>
      </svg>
    </div>
  );
};

// =============================================================================
// MINI CHART (Sparkline)
// =============================================================================

interface MiniChartProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
}

export const MiniChart: React.FC<MiniChartProps> = ({
  data,
  width = 100,
  height = 30,
  color,
}) => {
  if (!data || data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const isPositive = data[data.length - 1] >= data[0];
  const strokeColor = color || (isPositive ? '#4CAF50' : '#F44336');

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg width={width} height={height}>
      <polyline
        points={points}
        fill="none"
        stroke={strokeColor}
        strokeWidth="1.5"
      />
    </svg>
  );
};

export default CandlestickChart;
