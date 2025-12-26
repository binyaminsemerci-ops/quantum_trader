"""
Quantum Services Client - Integrates with all Quantum Trader microservices
"""
import aiohttp
import asyncio
import os
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class QuantumServicesClient:
    """Client for aggregating data from Quantum Trader services"""
    
    def __init__(self, timeout: int = 5):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
        # Use host.docker.internal for Docker, fallback to localhost for local dev
        service_host = os.getenv('QUANTUM_SERVICES_HOST', 'host.docker.internal')
        
        self.SERVICES = {
            'portfolio': f'http://{service_host}:8004',
            'trading': f'http://{service_host}:8003',
            'ai_engine': f'http://{service_host}:8001',
            'risk': f'http://{service_host}:8012',
            'strategy': f'http://{service_host}:8011',
            'ceo': f'http://{service_host}:8010',
            'model_supervisor': f'http://{service_host}:8007',
            'universe': f'http://{service_host}:8006',
            'backend': f'http://{service_host}:8000'
        }
    
    async def _get(self, service: str, endpoint: str) -> Optional[Dict[Any, Any]]:
        """Make GET request to a service"""
        base_url = self.SERVICES.get(service)
        if not base_url:
            logger.error(f"Unknown service: {service}")
            return None
        
        url = f"{base_url}{endpoint}"
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Service {service} returned {response.status}")
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout connecting to {service} at {url}")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"Error connecting to {service}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error with {service}: {e}")
            return None
    
    async def get_portfolio_summary(self) -> Optional[Dict[Any, Any]]:
        """Get portfolio summary from Portfolio Intelligence service"""
        return await self._get('portfolio', '/portfolio/summary')
    
    async def get_portfolio_positions(self) -> Optional[Dict[Any, Any]]:
        """Get current positions from Portfolio Intelligence"""
        return await self._get('portfolio', '/portfolio/positions')
    
    async def get_portfolio_performance(self) -> Optional[Dict[Any, Any]]:
        """Get portfolio performance metrics"""
        return await self._get('portfolio', '/portfolio/performance')
    
    async def get_live_trades(self) -> Optional[Dict[Any, Any]]:
        """Get active trades from Trading Bot"""
        return await self._get('trading', '/trades/active')
    
    async def get_trade_history(self, limit: int = 100) -> Optional[Dict[Any, Any]]:
        """Get trade history from Trading Bot"""
        return await self._get('trading', f'/trades/history?limit={limit}')
    
    async def get_trade_signals(self) -> Optional[Dict[Any, Any]]:
        """Get latest trading signals"""
        return await self._get('trading', '/signals/latest')
    
    async def get_ai_predictions(self) -> Optional[Dict[Any, Any]]:
        """Get AI model predictions"""
        return await self._get('ai_engine', '/predictions/latest')
    
    async def get_model_performance(self) -> Optional[Dict[Any, Any]]:
        """Get AI model performance metrics"""
        return await self._get('ai_engine', '/models/performance')
    
    async def get_confidence_scores(self) -> Optional[Dict[Any, Any]]:
        """Get model confidence scores"""
        return await self._get('ai_engine', '/confidence/scores')
    
    async def get_risk_metrics(self) -> Optional[Dict[Any, Any]]:
        """Get risk metrics from Risk Brain"""
        return await self._get('risk', '/risk/metrics')
    
    async def get_risk_var(self) -> Optional[Dict[Any, Any]]:
        """Get Value at Risk calculation"""
        return await self._get('risk', '/risk/var')
    
    async def get_risk_exposure(self) -> Optional[Dict[Any, Any]]:
        """Get current risk exposure"""
        return await self._get('risk', '/risk/exposure')
    
    async def get_strategy_performance(self) -> Optional[Dict[Any, Any]]:
        """Get strategy performance from Strategy Brain"""
        return await self._get('strategy', '/strategy/performance')
    
    async def get_model_health(self) -> Optional[Dict[Any, Any]]:
        """Get model health from Model Supervisor"""
        return await self._get('model_supervisor', '/models/health')
    
    async def get_market_universe(self) -> Optional[Dict[Any, Any]]:
        """Get market universe from Universe OS"""
        return await self._get('universe', '/universe/assets')
    
    async def health_check(self, service: str) -> bool:
        """Check if a service is healthy"""
        result = await self._get(service, '/health')
        return result is not None

# Global client instance
quantum_client = QuantumServicesClient()
