"""
Service RPC Layer - Redis Streams-based RPC for microservices
==============================================================

Provides RPC-like communication between microservices using Redis Streams
while maintaining compatibility with EventBus v2.

Features:
- Request/response pattern via Redis Streams
- Automatic timeout handling
- Exponential backoff retry
- Error propagation
- Distributed tracing integration

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 3.0.0
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Callable, Coroutine

from redis.asyncio import Redis

from backend.core.logger import get_logger

logger = get_logger(__name__, component="service_rpc")


# ============================================================================
# RPC DATA STRUCTURES
# ============================================================================

@dataclass
class RPCRequest:
    """RPC request structure"""
    trace_id: str
    service_target: str
    command: str
    parameters: Dict[str, Any]
    timeout_seconds: float
    timestamp: float
    source_service: str


@dataclass
class RPCResponse:
    """RPC response structure"""
    request_trace_id: str
    status: str  # SUCCESS, ERROR, TIMEOUT
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    error_code: Optional[str]
    processing_time_ms: float
    timestamp: float


# ============================================================================
# SERVICE RPC CLIENT
# ============================================================================

class ServiceRPCClient:
    """
    RPC client for inter-service communication via Redis Streams.
    
    Usage:
        rpc = ServiceRPCClient(redis_client, service_name="ai-service")
        await rpc.initialize()
        
        # Make RPC call
        response = await rpc.call(
            service="exec-risk-service",
            command="validate_risk",
            parameters={"symbol": "BTCUSDT", "size_usd": 1000}
        )
        
        if response.status == "SUCCESS":
            print(response.result)
    """
    
    STREAM_PREFIX = "quantum:rpc:"
    RESPONSE_TTL = 60  # Response stays in Redis for 60 seconds
    DEFAULT_TIMEOUT = 30.0  # 30 second default timeout
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0  # Exponential backoff base
    
    def __init__(
        self,
        redis_client: Redis,
        service_name: str,
        default_timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize RPC client.
        
        Args:
            redis_client: Async Redis client
            service_name: Name of this service
            default_timeout: Default RPC timeout in seconds
        """
        self.redis = redis_client
        self.service_name = service_name
        self.default_timeout = default_timeout
        
        # Response handlers: trace_id -> asyncio.Event + response
        self._pending_requests: Dict[str, asyncio.Event] = {}
        self._responses: Dict[str, RPCResponse] = {}
        
        # Response listener task
        self._listener_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(
            f"ServiceRPCClient initialized: service={service_name}, "
            f"timeout={default_timeout}s"
        )
    
    async def initialize(self) -> None:
        """Initialize RPC client and start response listener."""
        # Start response listener
        self._running = True
        self._listener_task = asyncio.create_task(self._response_listener())
        
        logger.info(f"ServiceRPCClient started: service={self.service_name}")
    
    async def shutdown(self) -> None:
        """Shutdown RPC client and cleanup."""
        self._running = False
        
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"ServiceRPCClient shutdown: service={self.service_name}")
    
    async def call(
        self,
        service: str,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        retries: int = 0,
    ) -> RPCResponse:
        """
        Make RPC call to another service.
        
        Args:
            service: Target service name ("ai-service", "exec-risk-service", etc.)
            command: Command to execute
            parameters: Command parameters
            timeout: Timeout in seconds (uses default if None)
            retries: Number of retries on timeout (0 = no retries)
        
        Returns:
            RPCResponse with result or error
        
        Raises:
            RuntimeError: If RPC client not initialized
        """
        if not self._running:
            raise RuntimeError("RPC client not initialized. Call initialize() first.")
        
        timeout = timeout or self.default_timeout
        parameters = parameters or {}
        
        # Create request
        trace_id = str(uuid.uuid4())
        request = RPCRequest(
            trace_id=trace_id,
            service_target=service,
            command=command,
            parameters=parameters,
            timeout_seconds=timeout,
            timestamp=time.time(),
            source_service=self.service_name,
        )
        
        # Try with retries
        for attempt in range(retries + 1):
            try:
                response = await self._execute_call(request, timeout)
                
                if response.status != "TIMEOUT" or attempt == retries:
                    return response
                
                # Exponential backoff before retry
                delay = self.RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(
                    f"RPC timeout, retrying in {delay}s (attempt {attempt + 1}/{retries + 1})",
                    trace_id=trace_id,
                    service=service,
                    command=command,
                )
                await asyncio.sleep(delay)
            
            except Exception as e:
                logger.error(
                    f"RPC call failed: {e}",
                    trace_id=trace_id,
                    service=service,
                    command=command,
                    exc_info=True,
                )
                return RPCResponse(
                    request_trace_id=trace_id,
                    status="ERROR",
                    result=None,
                    error_message=str(e),
                    error_code="RPC_CLIENT_ERROR",
                    processing_time_ms=0.0,
                    timestamp=time.time(),
                )
        
        # Should never reach here
        return RPCResponse(
            request_trace_id=trace_id,
            status="ERROR",
            result=None,
            error_message="Max retries exceeded",
            error_code="MAX_RETRIES_EXCEEDED",
            processing_time_ms=0.0,
            timestamp=time.time(),
        )
    
    async def _execute_call(self, request: RPCRequest, timeout: float) -> RPCResponse:
        """Execute single RPC call (internal)."""
        trace_id = request.trace_id
        
        # Register response handler
        response_event = asyncio.Event()
        self._pending_requests[trace_id] = response_event
        
        try:
            # Send request via Redis Stream
            await self._send_request(request)
            
            # Wait for response with timeout
            try:
                await asyncio.wait_for(response_event.wait(), timeout=timeout)
                
                # Get response
                response = self._responses.pop(trace_id, None)
                
                if response is None:
                    # Should not happen
                    logger.error(f"Response event set but no response found: {trace_id}")
                    return RPCResponse(
                        request_trace_id=trace_id,
                        status="ERROR",
                        result=None,
                        error_message="Internal error: response lost",
                        error_code="RESPONSE_LOST",
                        processing_time_ms=0.0,
                        timestamp=time.time(),
                    )
                
                return response
            
            except asyncio.TimeoutError:
                logger.warning(
                    f"RPC call timed out after {timeout}s",
                    trace_id=trace_id,
                    service=request.service_target,
                    command=request.command,
                )
                return RPCResponse(
                    request_trace_id=trace_id,
                    status="TIMEOUT",
                    result=None,
                    error_message=f"Timeout after {timeout}s",
                    error_code="RPC_TIMEOUT",
                    processing_time_ms=timeout * 1000,
                    timestamp=time.time(),
                )
        
        finally:
            # Cleanup
            self._pending_requests.pop(trace_id, None)
            self._responses.pop(trace_id, None)
    
    async def _send_request(self, request: RPCRequest) -> None:
        """Send RPC request via Redis Stream."""
        stream_name = self._get_request_stream(request.service_target)
        
        # Build message
        message = {
            "trace_id": request.trace_id,
            "source_service": request.source_service,
            "command": request.command,
            "parameters": json.dumps(request.parameters),
            "timeout_seconds": str(request.timeout_seconds),
            "timestamp": str(request.timestamp),
        }
        
        try:
            # XADD to target service stream
            message_id = await self.redis.xadd(stream_name, message, maxlen=1000)
            
            logger.debug(
                f"RPC request sent: {request.command}",
                trace_id=request.trace_id,
                service=request.service_target,
                message_id=message_id,
            )
        
        except Exception as e:
            logger.error(
                f"Failed to send RPC request: {e}",
                trace_id=request.trace_id,
                service=request.service_target,
            )
            raise
    
    async def _response_listener(self) -> None:
        """Background task that listens for RPC responses."""
        response_stream = self._get_response_stream(self.service_name)
        last_id = "0"
        
        logger.info(f"RPC response listener started: stream={response_stream}")
        
        while self._running:
            try:
                # XREAD - read new messages
                messages = await self.redis.xread(
                    {response_stream: last_id},
                    count=10,
                    block=5000,  # 5 second block
                )
                
                if not messages:
                    continue
                
                # Process messages
                for stream, msg_list in messages:
                    for message_id, message_data in msg_list:
                        await self._handle_response(message_data)
                        last_id = message_id
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in response listener: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        logger.info("RPC response listener stopped")
    
    async def _handle_response(self, message_data: Dict[bytes, bytes]) -> None:
        """Handle incoming RPC response."""
        try:
            # Decode message
            trace_id = message_data.get(b"trace_id", b"").decode("utf-8")
            status = message_data.get(b"status", b"ERROR").decode("utf-8")
            result_json = message_data.get(b"result", b"null").decode("utf-8")
            error_message = message_data.get(b"error_message", b"").decode("utf-8") or None
            error_code = message_data.get(b"error_code", b"").decode("utf-8") or None
            processing_time_ms = float(message_data.get(b"processing_time_ms", b"0").decode("utf-8"))
            timestamp = float(message_data.get(b"timestamp", b"0").decode("utf-8"))
            
            # Parse result
            result = None
            if result_json and result_json != "null":
                result = json.loads(result_json)
            
            # Create response
            response = RPCResponse(
                request_trace_id=trace_id,
                status=status,
                result=result,
                error_message=error_message,
                error_code=error_code,
                processing_time_ms=processing_time_ms,
                timestamp=timestamp,
            )
            
            # Check if anyone is waiting for this response
            if trace_id in self._pending_requests:
                self._responses[trace_id] = response
                self._pending_requests[trace_id].set()
                
                logger.debug(
                    f"RPC response received: {status}",
                    trace_id=trace_id,
                    processing_time_ms=processing_time_ms,
                )
            else:
                logger.warning(
                    f"Received response for unknown trace_id: {trace_id}",
                    status=status,
                )
        
        except Exception as e:
            logger.error(f"Failed to handle RPC response: {e}", exc_info=True)
    
    def _get_request_stream(self, service: str) -> str:
        """Get Redis Stream name for service requests."""
        return f"{self.STREAM_PREFIX}request:{service}"
    
    def _get_response_stream(self, service: str) -> str:
        """Get Redis Stream name for service responses."""
        return f"{self.STREAM_PREFIX}response:{service}"


# ============================================================================
# SERVICE RPC SERVER
# ============================================================================

class ServiceRPCServer:
    """
    RPC server for handling incoming RPC requests.
    
    Usage:
        server = ServiceRPCServer(redis_client, service_name="ai-service")
        
        # Register command handler
        @server.register_handler("get_signal")
        async def handle_get_signal(params: dict) -> dict:
            symbol = params["symbol"]
            signal = generate_signal(symbol)
            return {"action": signal.action, "confidence": signal.confidence}
        
        await server.start()
    """
    
    STREAM_PREFIX = "quantum:rpc:"
    MAX_PROCESSING_TIME = 30.0  # Max time for command handler
    
    def __init__(
        self,
        redis_client: Redis,
        service_name: str,
    ):
        """
        Initialize RPC server.
        
        Args:
            redis_client: Async Redis client
            service_name: Name of this service
        """
        self.redis = redis_client
        self.service_name = service_name
        
        # Command handlers: command -> async handler function
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]] = {}
        
        # Server task
        self._server_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"ServiceRPCServer initialized: service={service_name}")
    
    def register_handler(
        self,
        command: str,
    ) -> Callable:
        """
        Decorator for registering RPC command handler.
        
        Args:
            command: Command name
        
        Returns:
            Decorator function
        
        Example:
            @server.register_handler("validate_risk")
            async def handle_validate_risk(params: dict) -> dict:
                return {"approved": True, "max_size": 1000}
        """
        def decorator(handler: Callable) -> Callable:
            self._handlers[command] = handler
            logger.info(f"RPC handler registered: {command}")
            return handler
        return decorator
    
    async def start(self) -> None:
        """Start RPC server."""
        self._running = True
        self._server_task = asyncio.create_task(self._request_listener())
        
        logger.info(
            f"ServiceRPCServer started: service={self.service_name}, "
            f"handlers={len(self._handlers)}"
        )
    
    async def stop(self) -> None:
        """Stop RPC server."""
        self._running = False
        
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"ServiceRPCServer stopped: service={self.service_name}")
    
    async def _request_listener(self) -> None:
        """Background task that listens for RPC requests."""
        request_stream = self._get_request_stream(self.service_name)
        group_name = f"rpc_server_{self.service_name}"
        consumer_name = f"{self.service_name}_consumer"
        last_id = ">"
        
        # Create consumer group
        try:
            await self.redis.xgroup_create(request_stream, group_name, id="0", mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"Failed to create consumer group: {e}")
        
        logger.info(
            f"RPC request listener started: stream={request_stream}, "
            f"group={group_name}"
        )
        
        while self._running:
            try:
                # XREADGROUP - read new messages
                messages = await self.redis.xreadgroup(
                    group_name,
                    consumer_name,
                    {request_stream: last_id},
                    count=10,
                    block=5000,
                )
                
                if not messages:
                    continue
                
                # Process messages
                for stream, msg_list in messages:
                    for message_id, message_data in msg_list:
                        await self._handle_request(stream, group_name, message_id, message_data)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in request listener: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        logger.info("RPC request listener stopped")
    
    async def _handle_request(
        self,
        stream: bytes,
        group_name: str,
        message_id: bytes,
        message_data: Dict[bytes, bytes],
    ) -> None:
        """Handle incoming RPC request."""
        start_time = time.time()
        
        try:
            # Decode request
            trace_id = message_data.get(b"trace_id", b"").decode("utf-8")
            source_service = message_data.get(b"source_service", b"unknown").decode("utf-8")
            command = message_data.get(b"command", b"").decode("utf-8")
            parameters_json = message_data.get(b"parameters", b"{}").decode("utf-8")
            
            # Parse parameters
            parameters = json.loads(parameters_json)
            
            logger.debug(
                f"RPC request received: {command}",
                trace_id=trace_id,
                source=source_service,
            )
            
            # Check if handler exists
            handler = self._handlers.get(command)
            
            if not handler:
                # Unknown command
                await self._send_response(
                    source_service,
                    trace_id,
                    status="ERROR",
                    error_message=f"Unknown command: {command}",
                    error_code="UNKNOWN_COMMAND",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
                
                # Acknowledge message
                await self.redis.xack(stream, group_name, message_id)
                return
            
            # Execute handler with timeout
            try:
                result = await asyncio.wait_for(
                    handler(parameters),
                    timeout=self.MAX_PROCESSING_TIME,
                )
                
                # Send success response
                await self._send_response(
                    source_service,
                    trace_id,
                    status="SUCCESS",
                    result=result,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
            
            except asyncio.TimeoutError:
                logger.error(
                    f"RPC handler timeout: {command}",
                    trace_id=trace_id,
                )
                await self._send_response(
                    source_service,
                    trace_id,
                    status="ERROR",
                    error_message=f"Handler timeout after {self.MAX_PROCESSING_TIME}s",
                    error_code="HANDLER_TIMEOUT",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
            
            except Exception as e:
                logger.error(
                    f"RPC handler error: {e}",
                    trace_id=trace_id,
                    command=command,
                    exc_info=True,
                )
                await self._send_response(
                    source_service,
                    trace_id,
                    status="ERROR",
                    error_message=str(e),
                    error_code="HANDLER_ERROR",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
            
            # Acknowledge message
            await self.redis.xack(stream, group_name, message_id)
        
        except Exception as e:
            logger.error(f"Failed to handle RPC request: {e}", exc_info=True)
            
            # Try to acknowledge anyway
            try:
                await self.redis.xack(stream, group_name, message_id)
            except:
                pass
    
    async def _send_response(
        self,
        source_service: str,
        trace_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        """Send RPC response."""
        response_stream = self._get_response_stream(source_service)
        
        # Build message
        message = {
            "trace_id": trace_id,
            "status": status,
            "result": json.dumps(result) if result else "null",
            "error_message": error_message or "",
            "error_code": error_code or "",
            "processing_time_ms": str(processing_time_ms),
            "timestamp": str(time.time()),
        }
        
        try:
            # XADD to response stream
            await self.redis.xadd(response_stream, message, maxlen=1000)
            
            logger.debug(
                f"RPC response sent: {status}",
                trace_id=trace_id,
                target=source_service,
            )
        
        except Exception as e:
            logger.error(
                f"Failed to send RPC response: {e}",
                trace_id=trace_id,
                target=source_service,
            )
    
    def _get_request_stream(self, service: str) -> str:
        """Get Redis Stream name for service requests."""
        return f"{self.STREAM_PREFIX}request:{service}"
    
    def _get_response_stream(self, service: str) -> str:
        """Get Redis Stream name for service responses."""
        return f"{self.STREAM_PREFIX}response:{service}"
