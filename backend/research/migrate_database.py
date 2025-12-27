"""
Database migration for Strategy Generator AI tables.

Creates sg_strategies and sg_strategy_stats tables.
"""

from sqlalchemy import create_engine
from backend.database import DATABASE_URL, SessionLocal
from backend.research.schema import Base, Strategy, StrategyStatistics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate():
    """Create SG AI tables in database"""
    
    try:
        logger.info("=" * 60)
        logger.info("Strategy Generator AI - Database Migration")
        logger.info("=" * 60)
        
        logger.info(f"\nDatabase URL: {DATABASE_URL}")
        
        # Create engine
        from backend.database import engine
        
        # Create tables
        logger.info("\nCreating tables...")
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Created table: sg_strategies")
        logger.info("✅ Created table: sg_strategy_stats")
        
        # Verify tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        logger.info(f"\nVerifying tables...")
        
        if 'sg_strategies' in tables:
            logger.info("✅ sg_strategies table exists")
            
            # Show columns
            columns = [col['name'] for col in inspector.get_columns('sg_strategies')]
            logger.info(f"   Columns: {', '.join(columns)}")
        else:
            logger.error("❌ sg_strategies table not found")
        
        if 'sg_strategy_stats' in tables:
            logger.info("✅ sg_strategy_stats table exists")
            
            # Show columns
            columns = [col['name'] for col in inspector.get_columns('sg_strategy_stats')]
            logger.info(f"   Columns: {', '.join(columns)}")
        else:
            logger.error("❌ sg_strategy_stats table not found")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Migration complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    migrate()
