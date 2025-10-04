# Database Setup Guide

This guide explains how to set up and manage the Quantum Trader database using PostgreSQL and Alembic migrations.

## Database Architecture Decision

**Decision**: Use PostgreSQL with Alembic migrations for production-ready schema management.

**Rationale**:

- Better concurrent access handling for real-time trading
- Proper schema versioning with Alembic
- Production scalability for XGBoost training workloads
- Alignment with existing CI/CD pipeline (already configured for PostgreSQL)
- Better transaction handling and data integrity

## Quick Start

### Option 1: Docker Compose (Recommended for Development)

```bash
# Start PostgreSQL and all services
docker-compose up -d

# Run migrations
cd backend
alembic upgrade head

# Seed with demo data
python scripts/seed_demo_data.py
```

### Option 2: Local PostgreSQL Setup

1. **Install PostgreSQL** (version 12+)

2. **Create Database**:

```sql
createdb quantum_trader
```

1. **Set Environment Variables**:

```bash
export QUANTUM_TRADER_DATABASE_URL="postgresql://username:password@localhost/quantum_trader"
```

1. **Run Migrations**:

```bash
cd backend
pip install -r requirements.txt
alembic upgrade head
```

1. **Seed Demo Data**:

```bash
python scripts/seed_demo_data.py
```

### Option 3: Continue with SQLite (Development Only)

For local development, you can continue using SQLite:

```bash
cd backend
# SQLite is used by default when no DATABASE_URL is set
alembic upgrade head
python scripts/seed_demo_data.py
```

## Database Schema

Current schema includes:

### `trade_logs` table

- `id` (Primary Key)
- `symbol` (String, indexed) - Trading pair (e.g., "BTCUSDT")
- `side` (String) - "BUY" or "SELL"
- `qty` (Float) - Quantity traded
- `price` (Float) - Price per unit
- `status` (String) - "FILLED", "CANCELLED", "PARTIALLY_FILLED"
- `reason` (String, nullable) - Reason for trade decision
- `timestamp` (DateTime) - When trade occurred

### `settings` table  

- `id` (Primary Key)
- `api_key` (String) - Binance API key
- `api_secret` (String) - Binance API secret

## Migration Management

### Create New Migration

```bash
alembic revision --autogenerate -m "Description of changes"
```

### Apply Migrations

```bash
alembic upgrade head
```

### Rollback Migration

```bash
alembic downgrade -1  # Go back one migration
alembic downgrade base  # Reset to empty state
```

### Check Migration Status

```bash
alembic current
alembic history --verbose
```

## Environment Configuration

The database connection is configured via the `QUANTUM_TRADER_DATABASE_URL` environment variable:

### PostgreSQL (Production)

```bash
export QUANTUM_TRADER_DATABASE_URL="postgresql://user:pass@localhost:5432/quantum_trader"
```

### SQLite (Development)

```bash
# Leave unset to use default SQLite file
unset QUANTUM_TRADER_DATABASE_URL
```

### Docker Compose

The `docker-compose.yml` already configures PostgreSQL with:

- Host: `postgres` (internal Docker network)
- Port: `5432`
- Database: `quantum`
- User/Password: Set via environment variables

## Testing

The test suite uses a separate test database:

```bash
export QUANTUM_TRADER_DATABASE_URL="postgresql://test_user:test_pass@localhost/quantum_trader_test"
pytest backend/tests/
```

For SQLite testing:

```bash
export QUANTUM_TRADER_DATABASE_URL="sqlite:///test.db"
pytest backend/tests/
```

## Production Deployment

1. **Set up PostgreSQL instance** (AWS RDS, Google Cloud SQL, etc.)

2. **Configure connection string** with credentials:

```bash
export QUANTUM_TRADER_DATABASE_URL="postgresql://prod_user:prod_pass@db.example.com:5432/quantum_trader"
```

1. **Run migrations**:

```bash
alembic upgrade head
```

1. **Optional: Seed with production data**:

```bash
python scripts/seed_demo_data.py  # Only for demos
```

## Troubleshooting

### Common Issues

**Connection Refused**:

- Check PostgreSQL is running: `pg_isready`
- Verify connection string format
- Check firewall/network access

**Permission Denied**:

- Ensure database user has CREATE/ALTER privileges
- Check authentication method in `pg_hba.conf`

**Migration Conflicts**:

```bash
# Reset migrations (DESTRUCTIVE)
alembic downgrade base
alembic upgrade head
```

### Useful Commands

```bash
# Connect to database
psql $QUANTUM_TRADER_DATABASE_URL

# Backup database  
pg_dump $QUANTUM_TRADER_DATABASE_URL > backup.sql

# Restore database
psql $QUANTUM_TRADER_DATABASE_URL < backup.sql

# Check table sizes
psql $QUANTUM_TRADER_DATABASE_URL -c "\dt+"
```

## Future Schema Evolution

As the system grows, consider adding:

- `candles` table for OHLCV market data
- `signals` table for AI-generated trading signals  
- `model_registry` table for ML model versioning
- `user_sessions` table for authentication
- Proper foreign key relationships between tables

Each addition should be done via Alembic migrations to maintain schema consistency across environments.
