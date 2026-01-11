# AlphaTerminal Pro - AI Strategy System Architecture
## 7-Layer Institutional-Grade Machine Learning Pipeline

---

## 🎯 GENEL MİMARİ PRENSİPLERİ

### 1. Incremental Computation (Artımlı Hesaplama)
- Her tick/bar'da tüm feature'ları yeniden hesaplamak yerine **delta-based** güncelleme
- **Polars** kullanımı (Pandas'tan 10-100x hızlı)
- **Feature Store** (Redis) ile hesaplanmış değerlerin persist edilmesi
- **Streaming architecture** desteği

### 2. Causal Inference (Nedensellik)
- Korelasyon ≠ Nedensellik
- **SHAP (SHapley Additive exPlanations)** ile feature importance
- **Granger Causality** testleri
- **Counterfactual analysis** - "Bu feature olmasaydı ne olurdu?"

### 3. Statistically Robust Validation
- **Purged K-Fold Cross-Validation** (Lopez de Prado)
- **Embargo periods** - data leakage önleme
- **Combinatorial Purged Cross-Validation (CPCV)**
- **Walk-Forward Optimization** with anchored/rolling windows

### 4. Realistic Execution Modeling
- **Slippage simulation** (market impact)
- **Transaction costs** (spread + commission)
- **Latency modeling**
- **Fill probability** based on volume

### 5. Diversity & Regime Management
- **Strategy Zoo** - farklı piyasa rejimleri için strateji havuzu
- **Diversity metrics** - korelasyon bazlı çeşitlilik
- **Regime detection** - Bull/Bear/Sideways/High-Vol
- **Dynamic allocation** - rejime göre strateji ağırlıklandırma

---

## 📊 7-LAYER ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 7: EVOLUTION ENGINE                        │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐ │
│  │  Genetic    │ │  Strategy    │ │  Diversity  │ │  Retirement  │ │
│  │  Algorithm  │ │  Breeding    │ │  Manager    │ │  Manager     │ │
│  └─────────────┘ └──────────────┘ └─────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 6: LIVE EXECUTION                          │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐ │
│  │  Approval   │ │  Position    │ │  Slippage   │ │  Performance │ │
│  │  Checker    │ │  Sizer       │ │  Simulator  │ │  Monitor     │ │
│  └─────────────┘ └──────────────┘ └─────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 5: VALIDATION ENGINE                       │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐ │
│  │  Purged     │ │  Walk-Forward│ │  Monte Carlo│ │  Robustness  │ │
│  │  K-Fold CV  │ │  Analysis    │ │  Simulation │ │  Tests       │ │
│  └─────────────┘ └──────────────┘ └─────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 4: STRATEGY GENERATOR                      │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐ │
│  │  Rule       │ │  ML-Based    │ │  Hybrid     │ │  Risk        │ │
│  │  Synthesizer│ │  Generator   │ │  Composer   │ │  Calculator  │ │
│  └─────────────┘ └──────────────┘ └─────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 3: PATTERN DISCOVERY                       │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐ │
│  │  Decision   │ │  Clustering  │ │  SHAP       │ │  Granger     │ │
│  │  Tree Miner │ │  Engine      │ │  Explainer  │ │  Causality   │ │
│  └─────────────┘ └──────────────┘ └─────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: FEATURE FACTORY                         │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐ │
│  │  Technical  │ │  SMC         │ │  OrderFlow  │ │  Alpha       │ │
│  │  Indicators │ │  Features    │ │  Features   │ │  Features    │ │
│  │  (100+)     │ │  (50+)       │ │  (30+)      │ │  (20+)       │ │
│  └─────────────┘ └──────────────┘ └─────────────┘ └──────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Feature Store (Redis) + Incremental Calculator (Polars)    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: DATA LAYER                              │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐ │
│  │  Data       │ │  Quality     │ │  Regime     │ │  Universe    │ │
│  │  Fetcher    │ │  Checker     │ │  Detector   │ │  Manager     │ │
│  └─────────────┘ └──────────────┘ └─────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 LAYER 1: DATA LAYER (Veri Katmanı)

### Bileşenler:

#### 1.1 DataFetcher
```python
- Multi-source data aggregation (yfinance, APIs, WebSockets)
- Automatic retry with exponential backoff
- Rate limiting management
- Data normalization (OHLCV standardization)
```

#### 1.2 QualityChecker
```python
- Missing data detection & imputation
- Outlier detection (IQR, Z-score, MAD)
- Data consistency validation
- Corporate action adjustment (splits, dividends)
- Gap detection (trading halts, holidays)
```

#### 1.3 RegimeDetector
```python
- Hidden Markov Model (HMM) based regime detection
- Volatility regime (Low/Normal/High/Extreme)
- Trend regime (Bull/Bear/Sideways)
- Liquidity regime (Normal/Thin/Thick)
- Regime transition probabilities
```

#### 1.4 UniverseManager
```python
- Dynamic universe selection
- Liquidity filtering (min volume, min market cap)
- Sector rotation signals
- IPO/Delisting management
- Correlation-based universe optimization
```

---

## 🏭 LAYER 2: FEATURE FACTORY (Özellik Fabrikası)

### 2.1 Incremental Feature Engine (Polars-based)

```python
class IncrementalFeatureEngine:
    """
    Artımlı hesaplama ile 200+ feature.
    Her yeni bar geldiğinde sadece delta hesaplanır.
    """
    
    # Hesaplama modları
    FULL_RECALC = "full"      # Tüm history
    INCREMENTAL = "incremental"  # Sadece son bar
    WINDOWED = "windowed"     # Son N bar
```

### 2.2 Feature Kategorileri

#### A. Technical Indicators (100+ features)
```
TREND:
  - SMA (5, 10, 20, 50, 100, 200)
  - EMA (9, 12, 21, 26, 50)
  - DEMA, TEMA, KAMA
  - Supertrend
  - Ichimoku (Tenkan, Kijun, Senkou A/B, Chikou)
  - Parabolic SAR
  - ADX, DI+, DI-
  - Aroon Up/Down/Oscillator

MOMENTUM:
  - RSI (7, 14, 21)
  - Stochastic (%K, %D, SlowD)
  - Williams %R
  - CCI
  - MFI (Money Flow Index)
  - ROC (Rate of Change)
  - Momentum
  - TRIX
  - Ultimate Oscillator
  - Chande Momentum Oscillator

VOLATILITY:
  - ATR (7, 14, 21)
  - Bollinger Bands (Width, %B)
  - Keltner Channels
  - Donchian Channels
  - Standard Deviation
  - Historical Volatility
  - Parkinson Volatility
  - Garman-Klass Volatility
  - Yang-Zhang Volatility

VOLUME:
  - OBV (On Balance Volume)
  - Volume SMA ratios
  - VWAP
  - PVT (Price Volume Trend)
  - ADL (Accumulation/Distribution)
  - CMF (Chaikin Money Flow)
  - Force Index
  - EOM (Ease of Movement)
  - Volume Profile (POC, VAH, VAL)

CANDLESTICK PATTERNS:
  - Doji variations
  - Engulfing
  - Hammer/Hanging Man
  - Morning/Evening Star
  - Three White Soldiers/Black Crows
  - (50+ patterns via TA-Lib)
```

#### B. SMC Features (50+ features)
```
MARKET STRUCTURE:
  - swing_high_distance
  - swing_low_distance
  - structure_type (bullish/bearish/ranging)
  - bos_count_bullish
  - bos_count_bearish
  - choch_detected
  - higher_high_count
  - lower_low_count
  - structure_break_strength

ORDER BLOCKS:
  - bullish_ob_count
  - bearish_ob_count
  - nearest_bullish_ob_distance
  - nearest_bearish_ob_distance
  - ob_strength_score
  - ob_mitigation_rate
  - ob_volume_confirmation

FAIR VALUE GAPS:
  - bullish_fvg_count
  - bearish_fvg_count
  - fvg_fill_rate
  - largest_fvg_size_atr
  - fvg_proximity

LIQUIDITY:
  - buy_side_liquidity_distance
  - sell_side_liquidity_distance
  - liquidity_sweep_count
  - liquidity_grab_detected
  - stop_hunt_probability
```

#### C. OrderFlow Features (30+ features)
```
DELTA ANALYSIS:
  - delta
  - delta_percent
  - cumulative_delta
  - delta_divergence
  - delta_momentum

CVD (Cumulative Volume Delta):
  - cvd_value
  - cvd_trend
  - cvd_divergence
  - cvd_slope

ABSORPTION:
  - absorption_detected
  - absorption_strength
  - absorption_direction

IMBALANCE:
  - bid_ask_imbalance
  - volume_imbalance
  - trade_imbalance

INSTITUTIONAL:
  - large_trade_ratio
  - institutional_flow_score
  - smart_money_index
```

#### D. Alpha Features (20+ features)
```
PERFORMANCE:
  - jensen_alpha
  - alpha_vs_sector
  - alpha_vs_index
  - excess_return

RISK-ADJUSTED:
  - sharpe_ratio
  - sortino_ratio
  - calmar_ratio
  - information_ratio

RELATIVE STRENGTH:
  - rs_vs_sector
  - rs_vs_index
  - rs_percentile_rank

MOMENTUM:
  - momentum_1m
  - momentum_3m
  - momentum_6m
  - momentum_12m
  - acceleration
```

### 2.3 Feature Store (Redis-based)

```python
class FeatureStore:
    """
    Hesaplanmış feature'ların kalıcı depolanması.
    
    Key Pattern:
      features:{symbol}:{timeframe}:{feature_name}
      features:THYAO:4h:rsi_14 -> {"value": 65.5, "ts": 1704123456}
    
    Batch Pattern:
      features:{symbol}:{timeframe}:batch -> {feature1: val1, feature2: val2, ...}
    """
```

---

## 🔍 LAYER 3: PATTERN DISCOVERY (Örüntü Keşfi)

### 3.1 Decision Tree Miner
```python
class DecisionTreeMiner:
    """
    Kazanan trade'lerin ortak özelliklerini keşfeder.
    
    - Minimum leaf samples: 50 (overfitting önleme)
    - Max depth: 5 (interpretability)
    - Information Gain threshold: 0.1
    - Chi-square test for significance
    """
```

### 3.2 Clustering Engine
```python
class ClusteringEngine:
    """
    Piyasa koşullarını kümeleme.
    
    - K-Means for regime clustering
    - DBSCAN for anomaly detection
    - Hierarchical clustering for pattern taxonomy
    - Silhouette score for optimal k
    """
```

### 3.3 SHAP Explainer (Causal Inference)
```python
class SHAPExplainer:
    """
    Feature importance'ın nedensellik analizi.
    
    - TreeExplainer for tree-based models
    - KernelExplainer for any model
    - Feature interaction detection
    - Counterfactual analysis
    """
    
    def explain_strategy(self, strategy, trades):
        """
        Strateji başarısının hangi feature'lardan 
        kaynaklandığını açıklar.
        
        Returns:
            {
                "top_positive_features": [...],
                "top_negative_features": [...],
                "interaction_effects": [...],
                "counterfactual_scenarios": [...]
            }
        """
```

### 3.4 Granger Causality Tester
```python
class GrangerCausalityTester:
    """
    Feature -> Return nedensellik testi.
    
    - Lag optimization (1-10 periods)
    - Stationarity check (ADF test)
    - Multiple hypothesis correction (Bonferroni)
    """
```

---

## 🛠️ LAYER 4: STRATEGY GENERATOR (Strateji Üretici)

### 4.1 Rule Synthesizer
```python
class RuleSynthesizer:
    """
    Decision tree kurallarını trading stratejisine çevirir.
    
    Input: Tree path (RSI < 30 AND MACD_hist > 0 AND OB_proximity < 0.5)
    Output: TradingStrategy object with entry/exit rules
    """
```

### 4.2 ML-Based Generator
```python
class MLStrategyGenerator:
    """
    ML modelleri ile strateji üretimi.
    
    Models:
    - XGBoost for classification
    - LightGBM for speed
    - CatBoost for categorical features
    - Neural Network ensemble
    """
```

### 4.3 Risk Calculator
```python
class RiskCalculator:
    """
    Kurumsal seviye risk hesaplama.
    
    - Kelly Criterion (fractional)
    - Value at Risk (VaR 95%, 99%)
    - Expected Shortfall (CVaR)
    - Maximum Drawdown projection
    - Correlation-adjusted position sizing
    
    WORST-CASE SIMULATION:
    - Slippage: 0.1% - 0.5% (liquidity-dependent)
    - Spread: 0.05% - 0.2%
    - Commission: 0.1%
    - Failed fills: 5% probability
    """
```

---

## ✅ LAYER 5: VALIDATION ENGINE (Doğrulama Motoru)

### 5.1 Purged K-Fold Cross-Validation
```python
class PurgedKFoldCV:
    """
    Lopez de Prado'nun Purged K-Fold implementasyonu.
    
    Parameters:
    - n_splits: 5
    - purge_gap: max(feature_window, 20)  # Embargo period
    - embargo_pct: 0.01  # Additional safety margin
    
    Process:
    1. Split data into K folds
    2. Remove samples within purge_gap of test fold
    3. Additional embargo after test fold
    4. Train on remaining, test on fold
    """
```

### 5.2 Walk-Forward Analysis
```python
class WalkForwardAnalyzer:
    """
    Out-of-sample validation with rolling windows.
    
    Modes:
    - ANCHORED: Training window grows
    - ROLLING: Fixed training window moves
    - EXPANDING: Minimum window, then anchored
    
    Metrics per window:
    - Win rate
    - Profit factor
    - Sharpe ratio
    - Max drawdown
    
    Consistency Score:
    - % of windows with positive PnL
    - Std dev of window performances
    """
```

### 5.3 Monte Carlo Simulation
```python
class MonteCarloSimulator:
    """
    Trade sequence randomization.
    
    Simulations: 10,000
    
    Outputs:
    - Return distribution (mean, median, percentiles)
    - Drawdown distribution
    - VaR / CVaR at confidence levels
    - Probability of ruin
    - Time to recovery estimates
    """
```

### 5.4 Robustness Tests
```python
class RobustnessTestSuite:
    """
    Strateji dayanıklılık testleri.
    
    Tests:
    1. Parameter Sensitivity
       - ±10%, ±20% parameter variation
       - Performance stability check
    
    2. Time Period Stability
       - Different market regimes
       - Bull/Bear/Sideways subperiods
    
    3. Universe Stability
       - Random 80% subsample
       - Sector rotation
    
    4. Execution Assumptions
       - Slippage stress test
       - Delayed entry/exit
    """
```

---

## 🚀 LAYER 6: LIVE EXECUTION (Canlı Çalıştırma)

### 6.1 Approval Checker (Gateway)
```python
class ApprovalChecker:
    """
    Canlıya geçiş kontrol kapısı.
    
    MANDATORY CRITERIA:
    ├── Backtest win_rate >= 55%
    ├── Backtest profit_factor >= 1.5
    ├── Backtest Sharpe >= 1.0
    ├── Max drawdown <= 15%
    ├── Walk-forward consistency >= 60%
    ├── Monte Carlo VaR(95%) >= -10%
    └── Robustness score >= 0.7
    
    SOFT CRITERIA (warning only):
    ├── Expected profit > 3x spread cost
    ├── Average trade duration > 4 hours
    └── Sample size >= 100 trades
    
    Output: APPROVED / SANDBOX / REJECTED
    """
```

### 6.2 Position Sizer
```python
class PositionSizer:
    """
    Kelly-based position sizing with constraints.
    
    Calculation:
    1. Kelly fraction = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
    2. Fractional Kelly = Kelly * 0.25  # Conservative
    3. Volatility adjustment
    4. Correlation adjustment
    5. Maximum position cap
    
    Constraints:
    - Max 5% per position
    - Max 20% portfolio heat
    - Max 30% sector exposure
    - Max 50% correlated positions
    """
```

### 6.3 Slippage Simulator
```python
class SlippageSimulator:
    """
    Realistic execution modeling.
    
    Factors:
    - Order size / Average volume
    - Bid-ask spread
    - Market volatility
    - Time of day
    - Market impact model (Almgren-Chriss)
    """
```

### 6.4 Performance Monitor
```python
class PerformanceMonitor:
    """
    Real-time strateji performans takibi.
    
    Alerts:
    - 3 consecutive losses
    - Drawdown > 50% of max expected
    - Win rate deviation > 2 std
    - Sharpe < 50% of backtest
    
    Actions:
    - WARNING: Log and notify
    - PAUSE: Temporary halt
    - SANDBOX: Move to paper trading
    - RETIRE: Permanent deactivation
    """
```

---

## 🧬 LAYER 7: EVOLUTION ENGINE (Evrim Motoru)

### 7.1 Genetic Algorithm
```python
class GeneticAlgorithm:
    """
    Strateji parametrelerinin genetik optimizasyonu.
    
    Chromosome: Strategy parameters
    Fitness: Risk-adjusted return (Sharpe)
    
    Operators:
    - Selection: Tournament (k=3)
    - Crossover: Uniform (p=0.5)
    - Mutation: Gaussian (σ=0.1)
    
    Population: 100
    Generations: 50
    Elitism: Top 10%
    """
```

### 7.2 Strategy Breeding
```python
class StrategyBreeder:
    """
    Başarılı stratejilerin çaprazlanması.
    
    Process:
    1. Select top 2 parent strategies
    2. Extract successful rules from each
    3. Combine rules with compatibility check
    4. Validate child strategy
    5. Add to generation pool
    
    Diversity constraint:
    - Child must have <0.7 correlation with parents
    """
```

### 7.3 Diversity Manager (Strategy Zoo)
```python
class DiversityManager:
    """
    Strateji havuzu çeşitlilik yönetimi.
    
    Strategy Zoo Categories:
    ├── Bull Market Specialists
    ├── Bear Market Specialists
    ├── Sideways/Range Traders
    ├── High Volatility Plays
    ├── Low Volatility Plays
    ├── Sector Rotators
    └── All-Weather Strategies
    
    Diversity Metrics:
    - Return correlation matrix
    - Feature usage overlap
    - Trade timing overlap
    - Sector exposure overlap
    
    Target: Max 0.4 average pairwise correlation
    """
```

### 7.4 Retirement Manager
```python
class RetirementManager:
    """
    Düşük performanslı stratejilerin yönetimi.
    
    Monitoring Window: Last 10 signals
    
    Retirement Triggers:
    1. Win rate < 35% (over 10 trades)
    2. Actual Sharpe < 50% of MC expected Sharpe
    3. 5 consecutive losses
    4. Max drawdown exceeded
    5. Regime change (strategy's target regime ended)
    
    Retirement Process:
    1. ACTIVE → PROBATION (warning)
    2. PROBATION → SANDBOX (paper only)
    3. SANDBOX → RETIRED (archived)
    
    Revival Path:
    - If regime returns + backtest still valid
    - RETIRED → SANDBOX → PROBATION → ACTIVE
    """
```

---

## 📈 KPIs & MONITORING

### System KPIs
```
Strategy Generation:
- New strategies per day
- Approval rate
- Average backtest quality

Live Performance:
- Overall win rate
- Total PnL
- Sharpe ratio (realized)
- Max drawdown

Evolution:
- Generation count
- Diversity score
- Retirement rate
- Revival rate
```

### Alerts & Notifications
```
CRITICAL:
- System error
- Data feed failure
- All strategies paused

HIGH:
- Strategy retirement
- Unusual drawdown
- Regime change detected

MEDIUM:
- New strategy approved
- Performance deviation
- Diversity warning

LOW:
- New strategy generated
- Backtest completed
- Feature update
```

---

## 🔧 TECHNICAL STACK

```
Core:
- Python 3.11+
- Polars (data processing)
- NumPy, SciPy (numerical)
- Scikit-learn (ML)
- XGBoost, LightGBM, CatBoost (boosting)
- SHAP (explainability)
- Statsmodels (statistics)

Infrastructure:
- Redis (feature store, caching)
- PostgreSQL (strategy DB)
- Celery (task queue)
- APScheduler (scheduling)

Monitoring:
- Prometheus metrics
- Grafana dashboards
- Sentry (error tracking)
- Custom alerting
```

---

## 🚀 IMPLEMENTATION PHASES

### Phase 3A: Data Layer + Feature Factory (Files 1-10)
### Phase 3B: Pattern Discovery + Strategy Generator (Files 11-20)
### Phase 3C: Validation Engine (Files 21-25)
### Phase 3D: Live Execution (Files 26-30)
### Phase 3E: Evolution Engine (Files 31-35)
### Phase 3F: Integration & Testing (Files 36-40)

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: AlphaTerminal Team*
