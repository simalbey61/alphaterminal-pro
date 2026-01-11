# AlphaTerminal Pro v4.2

## 🚀 Enterprise-Grade BIST Trading Platform

Professional seviye hisse senedi analiz ve sinyal üretim platformu. Smart Money Concepts, OrderFlow analizi ve AI-powered strateji yönetimi.

---

## 📊 Platform Özeti

| Metrik | Değer |
|--------|-------|
| **Backend** | 90 Python dosyası, 39,223 satır kod |
| **Frontend** | 46 TypeScript dosyası |
| **Engine Sayısı** | 10 core engine |
| **API Endpoint** | 12+ RESTful endpoint |
| **Versiyon** | 4.2.0 |

---

## 🏗️ Mimari

```
┌─────────────────────────────────────────────────────────────────┐
│                     ALPHATERMINAL PRO v4.2                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  FRONTEND   │  │   NGINX     │  │        BACKEND          │  │
│  │   React     │──│   Proxy     │──│        FastAPI          │  │
│  │  TypeScript │  │   SSL/TLS   │  │        Python           │  │
│  └─────────────┘  └─────────────┘  └───────────┬─────────────┘  │
│                                                 │                │
│  ┌──────────────────────────────────────────────┴──────────────┐│
│  │                     CORE ENGINES                            ││
│  ├─────────────┬─────────────┬─────────────┬─────────────────┐ ││
│  │ SMC Engine  │ OrderFlow   │ Alpha       │ Risk Engine     │ ││
│  │ (1,869 LOC) │ (1,210 LOC) │ (729 LOC)   │ (1,270 LOC)     │ ││
│  ├─────────────┼─────────────┼─────────────┼─────────────────┤ ││
│  │ Correlation │ Audit       │ Shadow Mode │ Data Engine     │ ││
│  │ (921 LOC)   │ (1,144 LOC) │ (310 LOC)   │ (746 LOC)       │ ││
│  └─────────────┴─────────────┴─────────────┴─────────────────┘ ││
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   AI STRATEGY (7 Layers)                   │ │
│  │  Data → Features → Patterns → Evolution → Validation →    │ │
│  │        Strategy Generation → Live Execution               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ PostgreSQL  │  │   Redis     │  │   Telegram Bot          │ │
│  │ Database    │  │   Cache     │  │   Notifications         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Özellikler

### Core Engines
- **SMC Engine**: Smart Money Concepts (BOS, CHoCH, Order Blocks, FVG, Liquidity)
- **OrderFlow Engine**: Delta, CVD, VWAP, Footprint analizi
- **Alpha Engine**: Jensen's Alpha, Sharpe, Sortino, Momentum
- **Risk Engine**: Position sizing, drawdown control, portfolio heat
- **Correlation Engine**: Diversifikasyon, cluster analizi
- **Audit Engine**: Trade logging, compliance tracking

### AI Strategy System
- 7-katmanlı strateji geliştirme pipeline
- Genetik algoritma ile strateji optimizasyonu
- Walk-forward validation
- Monte Carlo simülasyonu
- Auto-retirement system

### Trading Features
- Multi-timeframe analiz
- Shadow Mode (paper trading)
- Telegram sinyal bildirimleri
- Real-time WebSocket updates
- Backtest engine

---

## 🛠️ Kurulum

### Gereksinimler
- Docker & Docker Compose
- Node.js 20+ (frontend geliştirme)
- Python 3.11+ (backend geliştirme)

### Hızlı Başlangıç

```bash
# 1. Repo'yu klonla
git clone https://github.com/your-org/alpha-terminal-pro.git
cd alpha-terminal-pro

# 2. Environment dosyasını oluştur
cp .env.example .env
# .env dosyasını düzenle

# 3. Docker ile başlat
make up

# veya
docker-compose up -d

# 4. Tarayıcıda aç
# Frontend: http://localhost:3000
# API: http://localhost:8000/docs
```

### Geliştirme Modu

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

---

## 📁 Proje Yapısı

```
alpha-terminal-pro/
├── backend/
│   ├── app/
│   │   ├── api/          # REST API endpoints
│   │   ├── core/         # Core engines (SMC, OrderFlow, etc.)
│   │   ├── ai_strategy/  # 7-layer AI system
│   │   ├── services/     # Business logic
│   │   ├── db/           # Database models & repos
│   │   ├── telegram/     # Bot integration
│   │   └── cache/        # Redis client
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Route pages
│   │   ├── store/        # Zustand store
│   │   ├── services/     # API client
│   │   └── hooks/        # Custom hooks
│   ├── package.json
│   └── Dockerfile
├── nginx/                # Reverse proxy config
├── scripts/              # DB init, utilities
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## 🔧 Konfigürasyon

### Environment Variables

```env
# Database
DB_PASSWORD=your-secure-password

# Security
SECRET_KEY=your-secret-key

# Telegram (Optional)
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
```

---

## 📈 API Endpoints

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/signals` | GET | Sinyal listesi |
| `/api/v1/signals/generate` | POST | Sinyal üret |
| `/api/v1/analysis/{symbol}` | GET | Hisse analizi |
| `/api/v1/portfolio` | GET | Portföy durumu |
| `/api/v1/strategies` | GET/POST | Strateji yönetimi |
| `/api/v1/backtest/run` | POST | Backtest çalıştır |

API dokümantasyonu: `http://localhost:8000/docs`

---

## 🧪 Test

```bash
# Backend tests
cd backend
pytest -v

# Frontend tests
cd frontend
npm test
```

---

## 📦 Deployment

### Production (Docker)

```bash
# Build & Deploy
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# SSL ile (Let's Encrypt)
docker-compose -f docker-compose.yml -f docker-compose.ssl.yml up -d
```

### Monitoring (Optional)

```bash
# Prometheus + Grafana
docker-compose --profile monitoring up -d
```

---

## 🔒 Güvenlik

- JWT tabanlı authentication
- Rate limiting
- CORS protection
- SQL injection prevention
- Input validation

---

## 📝 Lisans

Bu yazılım özel lisans altındadır. Ticari kullanım için izin alınmalıdır.

---

## 👥 Geliştirici

AlphaTerminal Team - 2024

---

## 🆘 Destek

Issues ve feature request'ler için GitHub Issues kullanın.

**⚠️ DİKKAT**: Bu yazılım finansal tavsiye vermez. Tüm yatırım kararları kendi sorumluluğunuzdadır.
