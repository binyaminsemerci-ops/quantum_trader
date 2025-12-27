"""
Postgres Connection Helper
SPRINT 3 - Module B: Postgres HA

Provides connection pooling and retry logic for Postgres connections.
"""

import time
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool, OperationalError

logger = logging.getLogger(__name__)


class PostgresConnectionPool:
    """
    Thread-safe Postgres connection pool with retry logic.
    
    Features:
    - Connection pooling (min/max connections)
    - Automatic retry with exponential backoff
    - Health checks
    - Connection timeout handling
    """
    
    def __init__(
        self,
        dsn: str,
        min_conn: int = 5,
        max_conn: int = 20,
        timeout: int = 10,
        max_retries: int = 3
    ):
        """
        Initialize connection pool.
        
        Args:
            dsn: Database connection string
                 Example: "postgresql://user:pass@host:5432/dbname"
            min_conn: Minimum connections to maintain
            max_conn: Maximum connections allowed
            timeout: Connection timeout in seconds
            max_retries: Maximum retry attempts on connection failure
        """
        self.dsn = dsn
        self.max_retries = max_retries
        self.timeout = timeout
        
        logger.info(f"Initializing Postgres connection pool (min={min_conn}, max={max_conn})")
        
        try:
            self.pool = pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                dsn=dsn,
                connect_timeout=timeout
            )
            logger.info("Connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def get_connection(self, retries: Optional[int] = None):
        """
        Get a connection from the pool with retry logic.
        
        Args:
            retries: Number of retry attempts (defaults to max_retries)
        
        Returns:
            psycopg2 connection object
        
        Raises:
            OperationalError: If connection fails after all retries
        """
        retries = retries or self.max_retries
        
        for attempt in range(1, retries + 1):
            try:
                conn = self.pool.getconn()
                
                # Test connection
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                
                logger.debug(f"Connection acquired from pool (attempt {attempt})")
                return conn
            
            except OperationalError as e:
                logger.warning(f"Connection failed (attempt {attempt}/{retries}): {e}")
                
                if attempt == retries:
                    logger.error("All connection attempts exhausted")
                    raise
                
                # Exponential backoff
                sleep_time = min(2 ** attempt, 30)  # Max 30 seconds
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        
        raise OperationalError("Failed to get connection after all retries")
    
    def return_connection(self, conn):
        """
        Return a connection to the pool.
        
        Args:
            conn: psycopg2 connection object
        """
        try:
            self.pool.putconn(conn)
            logger.debug("Connection returned to pool")
        except Exception as e:
            logger.error(f"Failed to return connection: {e}")
    
    @contextmanager
    def connection(self):
        """
        Context manager for safe connection handling.
        
        Usage:
            with pool.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trades")
                results = cursor.fetchall()
        """
        conn = None
        try:
            conn = self.get_connection()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Transaction failed: {e}")
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the connection pool.
        
        Returns:
            dict: Health status with connection metrics
        """
        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                
                # Query database version
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                
                # Query active connections
                cursor.execute("""
                    SELECT count(*) 
                    FROM pg_stat_activity 
                    WHERE state = 'active'
                """)
                active_connections = cursor.fetchone()[0]
                
                cursor.close()
            
            return {
                "status": "healthy",
                "version": version.split()[0],  # "PostgreSQL"
                "active_connections": active_connections,
                "pool_available": True
            }
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "pool_available": False
            }
    
    def close_all(self):
        """Close all connections in the pool."""
        try:
            self.pool.closeall()
            logger.info("All pool connections closed")
        except Exception as e:
            logger.error(f"Failed to close pool: {e}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Using context manager
    pool = PostgresConnectionPool(
        dsn="postgresql://postgres:password@localhost:5432/quantum_trader",
        min_conn=5,
        max_conn=20
    )
    
    # Query with automatic connection management
    with pool.connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, quantity FROM trades LIMIT 5")
        trades = cursor.fetchall()
        
        for symbol, qty in trades:
            print(f"{symbol}: {qty}")
    
    # Example 2: Manual connection handling
    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        print(f"Total trades: {count}")
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
    finally:
        pool.return_connection(conn)
    
    # Health check
    health = pool.health_check()
    print(f"Database health: {health}")
    
    # Cleanup
    pool.close_all()


# ============================================================================
# INTEGRATION WITH SERVICES
# ============================================================================

"""
# backend/services/execution/main.py

from infra.postgres.postgres_helper import PostgresConnectionPool

class ExecutionService:
    def __init__(self):
        self.db_pool = PostgresConnectionPool(
            dsn=os.getenv("DATABASE_URL"),
            min_conn=5,
            max_conn=20
        )
    
    async def save_trade(self, trade: Trade):
        with self.db_pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO trades (symbol, side, quantity, price) VALUES (%s, %s, %s, %s)",
                (trade.symbol, trade.side, trade.quantity, trade.price)
            )
"""
