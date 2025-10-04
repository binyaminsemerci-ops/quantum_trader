# 🚀 Quantum Trader

[![CI](https://github.com/binyaminsemerci-ops/quantum_trader/actions/workflows/ci.yml/badge.svg)](https://github.com/binyaminsemerci-ops/quantum_trader/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-4.x-blue.svg)](https://www.typescriptlang.org/)

**AI-Powered Cryptocurrency Trading Platform** with comprehensive analytics, real-time monitoring, and automated trading capabilities.

## ✨ Features

### 🧠 AI-Driven Trading
- **Machine Learning Signals** - XGBoost models for buy/sell predictions
- **Sentiment Analysis** - Twitter and news sentiment integration
- **Technical Analysis** - RSI, MACD, and custom indicators
- **Backtesting Engine** - Historical strategy validation

### 📊 Real-Time Analytics
- **Live Market Data** - Real-time price feeds and market depth
- **Performance Metrics** - Comprehensive trading analytics
- **Risk Management** - Position sizing and drawdown controls
- **Interactive Charts** - TradingView-style charting interface

### 🔄 Automated Trading
- **Exchange Integration** - Binance API support with extensible architecture
- **Order Management** - Smart order routing and execution
- **Portfolio Tracking** - Real-time P&L and position management
- **Alert System** - Configurable notifications and webhooks

### 🛠️ Developer Experience
- **RESTful API** - Comprehensive API with OpenAPI documentation
- **WebSocket Support** - Real-time data streaming
- **Performance Monitoring** - Built-in metrics and observability
- **Type Safety** - Full TypeScript coverage

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/binyaminsemerci-ops/quantum_trader.git
cd quantum_trader

# Start all services
docker-compose up --build

# Initialize database (first time only)
docker-compose exec backend alembic upgrade head
docker-compose exec backend python scripts/seed_demo_data.py
```

**Access the application:**
- 🖥️ **Frontend**: [http://localhost:3000](http://localhost:3000)
- 🔗 **API**: [http://localhost:8000](http://localhost:8000)
- 📚 **API Docs**: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

### Option 2: Local Development

**Backend Setup:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn main:app --reload
```

**Frontend Setup:**
```bash
cd frontend
npm install
npm run dev
```

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [📋 **API Documentation**](http://localhost:8000/api/docs) | Interactive OpenAPI/Swagger documentation |
| [🚀 **Deployment Guide**](./DEPLOYMENT.md) | Production deployment instructions |
| [🗄️ **Database Setup**](./DATABASE.md) | PostgreSQL and migration management |
| [🏗️ **Architecture**](./ARCHITECTURE.md) | System design and component overview |
| [🧪 **Testing Guide**](./backend/README.md) | Running tests and development setup |

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Frontend (React + TypeScript)"
        UI[Dashboard UI]
        Chart[Trading Charts]
        API_Client[API Client]
    end

    subgraph "Backend (FastAPI + Python)"
        Auth[Authentication]
        Trading[Trading Engine]
        ML[ML Models]
        WebSocket[WebSocket Server]
    end

    subgraph "External Services"
        Binance[Binance API]
        News[News APIs]
        Twitter[Twitter/X API]
    end

    subgraph "Infrastructure"
        DB[(PostgreSQL)]
        Redis[(Redis Cache)]
        Metrics[Monitoring]
    end

    UI --> API_Client
    Chart --> WebSocket
    API_Client --> Auth
    Auth --> Trading
    Trading --> ML
    Trading --> DB
    ML --> DB
    Trading --> Binance
    ML --> News
    ML --> Twitter
    WebSocket --> Redis
    Backend --> Metrics
```

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.12+)
- **Database**: PostgreSQL with Alembic migrations
- **ML/AI**: XGBoost, scikit-learn, pandas
- **APIs**: Binance, CryptoPanic, Twitter
- **Monitoring**: Custom performance metrics
- **Testing**: pytest with 58+ comprehensive tests

### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Charts**: Custom TradingView integration
- **State Management**: React Hooks + Context API
- **Testing**: Vitest

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Database**: PostgreSQL 16
- **Caching**: Redis (optional)
- **Monitoring**: Built-in performance metrics
- **CI/CD**: GitHub Actions

## 📊 Performance & Monitoring

Quantum Trader includes comprehensive performance monitoring:

- **Request Metrics** - Response times, throughput, error rates
- **Database Monitoring** - Query performance, connection pooling
- **System Resources** - CPU, memory, disk usage
- **Trading Metrics** - P&L, win rate, drawdown analysis

Access metrics via: [http://localhost:8000/api/metrics/requests](http://localhost:8000/api/metrics/requests)

## 🧪 Testing

The project maintains high test coverage with 58+ comprehensive tests:

```bash
# Backend tests
cd backend && pytest tests/ -v

# Frontend tests
cd frontend && npm run test

# All tests with coverage
cd backend && pytest tests/ --cov=. --cov-report=html
```

**Test Categories:**
- Unit tests for core trading logic
- Integration tests for API endpoints
- Performance monitoring tests
- Exception handling validation
- Database migration testing

## 🔧 Configuration

### Environment Variables

Create `.env` files based on the examples:

```bash
# Backend configuration
cp backend/.env.example backend/.env

# Frontend configuration
cp frontend/.env.example frontend/.env.local
```

### Key Configuration Options

```bash
# Database
QUANTUM_TRADER_DATABASE_URL=postgresql://user:pass@localhost/quantum_trader

# Trading APIs
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key

# External Data
CRYPTOPANIC_API_KEY=your_key
TWITTER_BEARER_TOKEN=your_token

# Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.

### Development Workflow

1. **Fork & Clone** the repository
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with tests
4. **Run the test suite**: `pytest tests/ && npm run test`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Code Quality

- **Type Safety**: Full TypeScript coverage
- **Testing**: Maintain >90% test coverage
- **Linting**: ESLint + Prettier for frontend, Black + isort for backend
- **Performance**: Monitor via built-in metrics

## 📈 Roadmap

### Current Status ✅
- [x] AI trading signals with XGBoost
- [x] Real-time data integration
- [x] Comprehensive API with OpenAPI docs
- [x] Performance monitoring system
- [x] PostgreSQL + Alembic migrations
- [x] 58+ comprehensive tests
- [x] Docker containerization

### Coming Next 🚧
- [ ] Advanced charting with indicators
- [ ] Multi-exchange support (Coinbase, Kraken)
- [ ] Mobile-responsive design improvements
- [ ] Advanced risk management features
- [ ] Real-time notifications system

### Future Vision 🔮
- [ ] Multi-asset support (stocks, forex, crypto)
- [ ] Social trading features
- [ ] Advanced portfolio optimization
- [ ] Mobile app development
- [ ] Institutional features

## 📄 License

This project is licensed under the [MIT License](./LICENSE) - see the license file for details.

## 🙋‍♂️ Support

- **📖 Documentation**: [API Docs](http://localhost:8000/api/docs) | [Deployment Guide](./DEPLOYMENT.md)
- **🐛 Issues**: [GitHub Issues](https://github.com/binyaminsemerci-ops/quantum_trader/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/binyaminsemerci-ops/quantum_trader/discussions)
- **📧 Contact**: [support@quantumtrader.dev](mailto:support@quantumtrader.dev)

---

**⚡ Built with passion for algorithmic trading and modern web technologies.**

*Join us in revolutionizing cryptocurrency trading through AI and superior developer experience.*
