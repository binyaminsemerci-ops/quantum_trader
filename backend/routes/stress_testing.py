"""
Stress Testing API - Scenario & Stress Testing System Integration

Provides endpoints for running stress tests, managing scenarios,
and retrieving stress test results.

Endpoints:
- POST /api/stress-testing/scenarios/run - Run single scenario
- POST /api/stress-testing/scenarios/batch - Run batch of scenarios
- GET /api/stress-testing/scenarios/library - List available scenarios
- GET /api/stress-testing/results/{scenario_name} - Get scenario result
- GET /api/stress-testing/status - Get system status
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field

from backend.services.stress_testing import (
    Scenario,
    ScenarioType,
    ScenarioResult,
    ScenarioExecutor,
    ScenarioLibrary,
    StressTestRunner
)
from backend.services.stress_testing.quantum_trader_adapter import (
    create_quantum_trader_executor
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stress-testing", tags=["stress-testing"])

# Global state for storing results
_stress_test_results: Dict[str, Dict[str, Any]] = {}
_stress_test_status: Dict[str, str] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class ScenarioRequest(BaseModel):
    """Request to run a scenario"""
    name: str
    type: str  # ScenarioType enum value
    parameters: Dict[str, Any] = Field(default_factory=dict)
    symbols: Optional[List[str]] = None
    start: Optional[str] = None  # ISO datetime
    end: Optional[str] = None    # ISO datetime
    seed: Optional[int] = None


class BatchScenarioRequest(BaseModel):
    """Request to run batch of scenarios"""
    scenario_names: List[str]  # Names from library
    parallel: bool = True
    max_workers: int = 4


class ScenarioResultResponse(BaseModel):
    """Scenario result summary"""
    scenario_name: str
    success: bool
    duration_sec: float
    
    # Performance metrics
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    winrate: float
    
    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # System health
    emergency_stops: int
    execution_failures: int
    failed_models: List[str]
    failed_strategies: List[str]
    
    # Curves (truncated for API)
    equity_curve_sample: List[float] = Field(default_factory=list)
    pnl_curve_sample: List[float] = Field(default_factory=list)
    
    notes: str = ""


class BatchResultResponse(BaseModel):
    """Batch test results"""
    total_scenarios: int
    successful: int
    failed: int
    total_duration_sec: float
    
    # Aggregate stats
    total_trades: int
    avg_winrate: float
    avg_max_drawdown: float
    worst_drawdown: float
    scenarios_with_ess: int
    
    # Individual results
    results: List[ScenarioResultResponse]


class LibraryScenario(BaseModel):
    """Library scenario info"""
    name: str
    type: str
    description: str
    parameters: Dict[str, Any]


class SystemStatus(BaseModel):
    """Stress testing system status"""
    available: bool
    executor_initialized: bool
    running_tests: int
    completed_tests: int
    last_run: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def _scenario_result_to_response(name: str, result: ScenarioResult) -> ScenarioResultResponse:
    """Convert ScenarioResult to API response"""
    # Sample curves (return every 10th point to reduce payload)
    equity_sample = result.equity_curve[::10] if result.equity_curve else []
    pnl_sample = result.pnl_curve[::10] if result.pnl_curve else []
    
    return ScenarioResultResponse(
        scenario_name=name,
        success=result.success,
        duration_sec=result.duration_sec,
        total_pnl=result.total_pnl,
        total_pnl_pct=result.total_pnl_pct,
        max_drawdown=result.max_drawdown,
        sharpe_ratio=result.sharpe_ratio,
        profit_factor=result.profit_factor,
        winrate=result.winrate,
        total_trades=result.total_trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        emergency_stops=result.emergency_stops,
        execution_failures=result.execution_failures,
        failed_models=result.failed_models,
        failed_strategies=result.failed_strategies,
        equity_curve_sample=equity_sample[:100],  # Max 100 points
        pnl_curve_sample=pnl_sample[:100],
        notes=result.notes
    )


def _create_scenario_from_request(req: ScenarioRequest) -> Scenario:
    """Create Scenario object from API request"""
    try:
        scenario_type = ScenarioType(req.type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario type: {req.type}. Must be one of: {[t.value for t in ScenarioType]}"
        )
    
    # Parse datetime strings
    start = datetime.fromisoformat(req.start) if req.start else None
    end = datetime.fromisoformat(req.end) if req.end else None
    
    return Scenario(
        name=req.name,
        type=scenario_type,
        parameters=req.parameters,
        symbols=req.symbols,
        start=start,
        end=end,
        seed=req.seed
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/status", response_model=SystemStatus)
async def get_status():
    """Get stress testing system status"""
    return SystemStatus(
        available=True,
        executor_initialized=True,  # TODO: Check actual state
        running_tests=len([s for s in _stress_test_status.values() if s == "running"]),
        completed_tests=len(_stress_test_results),
        last_run=max(_stress_test_results.keys()) if _stress_test_results else None
    )


@router.get("/scenarios/library", response_model=List[LibraryScenario])
async def list_library_scenarios():
    """List all available scenarios from the library"""
    scenarios = ScenarioLibrary.get_all()
    
    response = []
    for scenario in scenarios:
        response.append(LibraryScenario(
            name=scenario.name,
            type=scenario.type.value,
            description=f"{scenario.type.value} scenario: {scenario.name}",
            parameters=scenario.parameters
        ))
    
    return response


@router.post("/scenarios/run")
async def run_scenario(
    scenario_request: ScenarioRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Run a single stress test scenario.
    
    Returns immediately with a task ID. Use /results/{scenario_name} to get results.
    """
    scenario = _create_scenario_from_request(scenario_request)
    
    # Mark as running
    _stress_test_status[scenario.name] = "running"
    
    async def run_test():
        """Background task to run the test"""
        try:
            # Create executor with real Quantum Trader components
            executor = create_quantum_trader_executor(
                app_state=request.app.state,
                initial_capital=100000.0
            )
            
            runner = StressTestRunner(executor=executor, max_workers=1)
            results = runner.run_batch([scenario], parallel=False)
            
            # Store result
            result = results[scenario.name]
            _stress_test_results[scenario.name] = {
                "scenario": scenario,
                "result": result,
                "completed_at": datetime.utcnow().isoformat()
            }
            _stress_test_status[scenario.name] = "completed"
            
            logger.info(f"[STRESS-TEST] Completed: {scenario.name}")
            
        except Exception as e:
            logger.error(f"[STRESS-TEST] Failed: {scenario.name} - {e}")
            _stress_test_status[scenario.name] = "failed"
            _stress_test_results[scenario.name] = {
                "scenario": scenario,
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat()
            }
    
    # Schedule background task
    background_tasks.add_task(run_test)
    
    return {
        "status": "submitted",
        "scenario_name": scenario.name,
        "message": f"Stress test '{scenario.name}' started. Check /results/{scenario.name} for results."
    }


@router.post("/scenarios/batch", response_model=BatchResultResponse)
async def run_batch_scenarios(batch_request: BatchScenarioRequest, request: Request):
    """
    Run batch of scenarios from the library.
    
    This is a synchronous endpoint - waits for all tests to complete.
    Use for small batches or when you need immediate results.
    """
    # Get scenarios from library
    library_scenarios = ScenarioLibrary.get_all()
    library_map = {s.name: s for s in library_scenarios}
    
    scenarios = []
    for name in batch_request.scenario_names:
        if name not in library_map:
            raise HTTPException(
                status_code=404,
                detail=f"Scenario '{name}' not found in library"
            )
        scenarios.append(library_map[name])
    
    # Create executor with real Quantum Trader components
    executor = create_quantum_trader_executor(
        app_state=request.app.state,
        initial_capital=100000.0
    )
    
    # Run batch
    runner = StressTestRunner(executor=executor, max_workers=batch_request.max_workers)
    
    logger.info(f"[STRESS-TEST] Running batch: {len(scenarios)} scenarios")
    start_time = datetime.utcnow()
    
    results = runner.run_batch(scenarios, parallel=batch_request.parallel)
    summary = runner.generate_summary(results)
    
    duration = (datetime.utcnow() - start_time).total_seconds()
    
    # Store results
    for name, result in results.items():
        _stress_test_results[name] = {
            "scenario": library_map[name],
            "result": result,
            "completed_at": datetime.utcnow().isoformat()
        }
    
    # Convert to response
    result_responses = [
        _scenario_result_to_response(name, result)
        for name, result in results.items()
    ]
    
    return BatchResultResponse(
        total_scenarios=summary["total_scenarios"],
        successful=summary["successful"],
        failed=summary["failed"],
        total_duration_sec=duration,
        total_trades=summary["total_trades"],
        avg_winrate=summary["avg_winrate"],
        avg_max_drawdown=summary["avg_max_drawdown"],
        worst_drawdown=summary["worst_drawdown"],
        scenarios_with_ess=summary["total_emergency_stops"],
        results=result_responses
    )


@router.get("/results/{scenario_name}", response_model=ScenarioResultResponse)
async def get_scenario_result(scenario_name: str):
    """Get results for a specific scenario"""
    if scenario_name not in _stress_test_results:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for scenario '{scenario_name}'"
        )
    
    stored = _stress_test_results[scenario_name]
    
    # Check if it's an error result
    if "error" in stored:
        raise HTTPException(
            status_code=500,
            detail=f"Scenario failed: {stored['error']}"
        )
    
    result = stored["result"]
    return _scenario_result_to_response(scenario_name, result)


@router.get("/results")
async def list_results():
    """List all available results"""
    return {
        "count": len(_stress_test_results),
        "scenarios": [
            {
                "name": name,
                "status": _stress_test_status.get(name, "completed"),
                "completed_at": stored.get("completed_at"),
                "has_error": "error" in stored
            }
            for name, stored in _stress_test_results.items()
        ]
    }


@router.delete("/results/{scenario_name}")
async def delete_result(scenario_name: str):
    """Delete a scenario result"""
    if scenario_name not in _stress_test_results:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for scenario '{scenario_name}'"
        )
    
    del _stress_test_results[scenario_name]
    if scenario_name in _stress_test_status:
        del _stress_test_status[scenario_name]
    
    return {"status": "deleted", "scenario_name": scenario_name}


@router.delete("/results")
async def clear_all_results():
    """Clear all stored results"""
    count = len(_stress_test_results)
    _stress_test_results.clear()
    _stress_test_status.clear()
    
    return {"status": "cleared", "deleted_count": count}
