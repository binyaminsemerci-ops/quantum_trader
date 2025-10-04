# Database Scripts

This directory contains database management and seeding scripts for Quantum Trader.

## Available Scripts

### `seed_demo_data.py`
Populates the database with sample trade data for demo and development purposes.

**Usage**:
```bash
cd backend
python scripts/seed_demo_data.py
```

**What it creates**:
- Sample trades with different statuses (FILLED, CANCELLED, PARTIALLY_FILLED)
- Demo API settings for development
- Realistic timestamps and trading data

**Requirements**:
- Database must be set up with `alembic upgrade head` first
- Appropriate DATABASE_URL configuration

The script is safe to run multiple times - it will skip creating settings if they already exist.
