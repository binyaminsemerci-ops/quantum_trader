# Copilot Instructions for Quantum Trader

This document provides context and guidelines for GitHub Copilot when working on the Quantum Trader project.

## Project Overview

Quantum Trader is an AI-powered cryptocurrency trading platform featuring:
- **Backend**: FastAPI (Python 3.12+) with PostgreSQL database
- **Frontend**: React 18 with TypeScript, Vite, and Tailwind CSS
- **AI/ML**: XGBoost models for trading signals, sentiment analysis
- **APIs**: Binance integration, CryptoPanic, Twitter
- **Infrastructure**: Docker, Docker Compose, GitHub Actions CI/CD

## Architecture

- **Backend**: FastAPI application with SQLAlchemy ORM, Alembic migrations
- **Frontend**: React SPA with TypeScript, Chart.js for visualizations
- **Database**: PostgreSQL 16 with Alembic for schema management
- **Testing**: pytest (backend with 58+ tests), Vitest (frontend)
- **Monitoring**: Custom performance metrics and logging

## Coding Standards

### Backend (Python)

- **Python Version**: 3.12+
- **Style**: Follow PEP 8, use type hints
- **Linting**: Use `ruff check backend` and `mypy backend`
- **Imports**: Use absolute imports from backend root
- **Testing**: Write pytest tests for new features
- **Async**: Use FastAPI's async/await patterns where appropriate
- **Error Handling**: Use custom exceptions from `backend/exceptions.py`
- **Configuration**: Read from environment variables (see `.env.example`)
- **Security**: Never commit API keys or credentials

### Frontend (TypeScript/React)

- **TypeScript**: Full type coverage, strict mode enabled
- **Style**: ESLint + Prettier for consistency
- **Components**: Functional components with hooks
- **State Management**: React Hooks + Context API
- **API Calls**: Use axios with type-safe API client
- **Testing**: Write Vitest tests with React Testing Library
- **Styling**: Tailwind CSS utility classes

## Testing Requirements

### Backend Tests
```bash
# Run all tests
python -m pytest -q

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

- Maintain >90% test coverage
- Write unit tests for business logic
- Use fixtures for database tests
- Mock external API calls (ccxt, exchanges)

### Frontend Tests
```bash
# Run tests
npm run test

# Type checking
npm run typecheck
```

- Test components with React Testing Library
- Mock API responses
- Test error handling and edge cases

## Key Patterns & Conventions

### Backend

1. **Routes**: Organize by feature in `backend/routes/`
2. **Models**: SQLAlchemy models in `backend/models/`
3. **Database**: Use Alembic migrations for schema changes
4. **Config**: Use `backend/config.py` for settings
5. **Logging**: Import from `backend/logging_config.py`
6. **Performance**: Use `backend/performance_monitor.py` for metrics

### Frontend

1. **Components**: Organize in `frontend/src/components/`
2. **API Client**: Use centralized API client
3. **Types**: Define interfaces for API responses
4. **Error Boundaries**: Handle errors gracefully
5. **Loading States**: Show appropriate feedback to users

## Dependencies

### Backend Core
- FastAPI, uvicorn
- SQLAlchemy, Alembic
- PostgreSQL (psycopg2-binary)
- XGBoost, scikit-learn, pandas
- python-binance for exchange API
- python-dotenv for environment variables

### Frontend Core
- React 18, TypeScript 5
- Vite for build tooling
- Tailwind CSS for styling
- axios for HTTP requests
- Chart.js for data visualization
- Vitest for testing

## Development Workflow

1. **Branching**: Create feature branches: `feat/<name>`, `fix/<name>`, `chore/<name>`
2. **Commits**: Write meaningful commit messages
3. **Testing**: All tests must pass before PR
4. **Linting**: Run linters before committing
5. **Documentation**: Update docs for significant changes
6. **Security**: Review for secrets before committing

## Pre-PR Checklist

- [ ] All tests pass: `python -m pytest -q` (backend) and `npm run test` (frontend)
- [ ] Linters pass: `python -m ruff check backend` and `python -m mypy backend`
- [ ] No secrets in code (check `.env.example`)
- [ ] Update `requirements.txt` if dependencies changed
- [ ] Add tests for new functionality
- [ ] Update documentation if needed

## Common Commands

### Backend
```bash
# Run development server
cd backend && uvicorn main:app --reload

# Run migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"
```

### Frontend
```bash
# Run development server
cd frontend && npm run dev

# Build for production
npm run build

# Preview production build
npm run serve
```

### Docker
```bash
# Start all services
docker-compose up --build

# Initialize database
docker-compose exec backend alembic upgrade head
docker-compose exec backend python scripts/seed_demo_data.py
```

## Important Files

- `CONTRIBUTING.md` - Contribution guidelines
- `README_NEW.md` - Main project documentation
- `ARCHITECTURE.md` - System architecture
- `DEPLOYMENT.md` - Deployment guide
- `DATABASE.md` - Database setup
- `.env.example` - Environment variable template
- `backend/main.py` - FastAPI application entry point
- `frontend/src/main.tsx` - React application entry point

## Security Considerations

- Never commit API keys, passwords, or tokens
- Use environment variables for sensitive data
- Validate and sanitize all user inputs
- Use HTTPS for production deployments
- Review security implications for trading operations
- For sensitive changes (secrets, model loading, order execution), request additional review

## AI/ML Specific

- Training artifacts stored in `ai_engine/models/`
- Use `train_ai_model.bat` (Windows) or equivalent for training
- Models: XGBoost for predictions
- Features: Technical indicators (RSI, MACD) + sentiment analysis
- Backtesting available for strategy validation

## Performance & Monitoring

- Use structured logging via `backend/logging_config.py`
- Track metrics with `backend/performance_monitor.py`
- Monitor endpoints: `/api/metrics/requests`, `/api/metrics/db`, `/api/metrics/system`
- Optimize database queries (use indexes, avoid N+1)
- Cache frequently accessed data where appropriate

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)

---

For detailed contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).
