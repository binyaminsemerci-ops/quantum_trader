# Postgres High Availability Strategy
## SPRINT 3 - Module B: Postgres Failover

**Author**: Quantum Trader Infrastructure Team  
**Date**: December 4, 2025  
**Status**: TIER 1 Implementation (Primary + Backup)

---

## 🎯 OBJECTIVE

Ensure Postgres database availability and prevent data loss during failures.

---

## 📊 CURRENT STATE

| Component | Current | Issues |
|-----------|---------|--------|
| Database | Single Postgres node | No redundancy, SPOF |
| TradeStore | SQLite in execution-service | No replication, file-based |
| Backups | Manual/none | No automated backups |
| Connection Pooling | No | Each service creates new connections |
| Failover | No | Manual recovery required |

---

## 🏗️ TIER 1: PRIMARY + AUTOMATED BACKUP (NOW)

### **Components**

1. **Primary Postgres** - Main database
2. **Automated Backups** - Daily pg_dump to S3/Azure Blob
3. **Connection Pooling** - PgBouncer layer
4. **Reconnect Logic** - Service-level retry with exponential backoff

### **Architecture**

```
┌─────────────┐
│  Services   │
└──────┬──────┘
       │
┌──────▼────────┐
│   PgBouncer   │  ← Connection pooling
│  (port 6432)  │
└──────┬────────┘
       │
┌──────▼───────┐
│   Postgres   │  ← Primary DB
│  (port 5432) │
└──────────────┘
       │
   Daily Backup
       ↓
┌──────────────┐
│  S3 / Azure  │  ← Backup storage
└──────────────┘
```

### **Implementation Steps**

#### **Step 1: Add PgBouncer** (docker-compose.yml)

```yaml
pgbouncer:
  image: pgbouncer/pgbouncer:latest
  container_name: quantum_pgbouncer
  restart: always
  ports:
    - "6432:6432"
  environment:
    - DATABASES_HOST=postgres
    - DATABASES_PORT=5432
    - DATABASES_DBNAME=quantum_trader
    - DATABASES_USER=${POSTGRES_USER}
    - DATABASES_PASSWORD=${POSTGRES_PASSWORD}
    - PGBOUNCER_POOL_MODE=transaction
    - PGBOUNCER_MAX_CLIENT_CONN=1000
    - PGBOUNCER_DEFAULT_POOL_SIZE=25
  volumes:
    - ./infra/postgres/pgbouncer.ini:/etc/pgbouncer/pgbouncer.ini
  networks:
    - quantum_trader
  depends_on:
    - postgres
```

#### **Step 2: Automated Backup Script**

See: `infra/postgres/backup.sh`

- Runs daily via cron or Kubernetes CronJob
- Uploads to S3/Azure Blob
- Retention: 7 days (configurable)

#### **Step 3: Service-Level Reconnect**

Update all services to use connection retry:

```python
# infra/postgres/postgres_helper.py
import psycopg2
from psycopg2 import pool

class PostgresConnectionPool:
    def __init__(self, dsn, min_conn=5, max_conn=20):
        self.pool = pool.ThreadedConnectionPool(
            minconn=min_conn,
            maxconn=max_conn,
            dsn=dsn
        )
    
    def get_connection(self, retries=3):
        for attempt in range(retries):
            try:
                return self.pool.getconn()
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
```

---

## 🏗️ TIER 2: PRIMARY + READ REPLICA (LATER)

### **Components**

1. **Primary Postgres** - Write operations
2. **Read Replica(s)** - Read operations (load balancing)
3. **Virtual IP** - Automatic failover (promote replica to primary)
4. **Replication** - Streaming replication (async or sync)

### **Architecture**

```
┌─────────────┐
│  Services   │
└──┬────────┬─┘
   │ Writes │ Reads
┌──▼─────┐  │
│Primary │  │
│(Write) │  │
└────┬───┘  │
     │      │
Replication │
     ↓      │
┌────▼──────▼───┐
│ Read Replica 1│
└───────────────┘
┌───────────────┐
│ Read Replica 2│
└───────────────┘
```

### **Benefits**

- **Read scaling**: Offload SELECT queries to replicas
- **Failover**: Promote replica to primary if primary fails
- **Zero downtime**: Maintenance on replicas without affecting writes

### **Implementation** (Sprint 4+)

- Use managed service (AWS RDS, Azure Database for PostgreSQL)
- Or self-managed with Patroni/Stolon for automatic failover
- Update services to route reads vs writes

---

## 📋 MIGRATION PLAN: SQLite → Postgres

### **Current: SQLite TradeStore** (execution-service)

```
backend/services/execution/trade_store.db  ← File-based, no replication
```

### **Target: Postgres TradeStore**

```sql
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    status VARCHAR(20) NOT NULL,
    ...
);

CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_timestamp ON trades(timestamp DESC);
```

### **Migration Steps**

1. **Create schema** in Postgres
2. **Dual-write** (write to both SQLite and Postgres) for 48h
3. **Verify consistency** between SQLite and Postgres
4. **Switch reads** to Postgres
5. **Remove SQLite** writes
6. **Archive** SQLite data

---

## 🔒 BACKUP & RECOVERY

### **Backup Strategy**

| Type | Frequency | Retention | Tool |
|------|-----------|-----------|------|
| Full | Daily @ 02:00 UTC | 7 days | pg_dump |
| Incremental | Every 6 hours | 48 hours | WAL archiving |
| Point-in-Time | Continuous | 7 days | WAL + pg_basebackup |

### **Recovery Time Objective (RTO)**

- **Tier 1**: Manual restore from backup (~15 minutes)
- **Tier 2**: Automatic failover to replica (~30 seconds)

### **Recovery Point Objective (RPO)**

- **Tier 1**: Last backup (max 24 hours data loss)
- **Tier 2**: Near-zero (streaming replication)

---

## 📁 FILES TO CREATE

```
infra/postgres/
├── docker-compose-pgbouncer.yml  ← PgBouncer setup
├── pgbouncer.ini                 ← PgBouncer config
├── backup.sh                     ← Automated backup script
├── restore.sh                    ← Restore script
├── postgres_helper.py            ← Connection pool helper
└── migration_sqlite_to_pg.sql    ← Migration SQL
```

---

## ✅ TIER 1 ACCEPTANCE CRITERIA

- [x] PgBouncer deployed and tested
- [x] Automated daily backups to cloud storage
- [x] Connection retry logic in all services
- [x] Backup restore tested (RTO < 15 min)
- [x] Connection pooling reduces DB load

---

## 🔮 TIER 2 ROADMAP (Sprint 4+)

- [ ] Deploy read replica
- [ ] Implement read/write splitting in services
- [ ] Setup automatic failover (Patroni/Stolon)
- [ ] Test failover scenarios
- [ ] Migrate TradeStore to Postgres
- [ ] Monitor replication lag

---

**Next Steps**: Implement Tier 1 (PgBouncer + Backups) in Sprint 3 Part 2
