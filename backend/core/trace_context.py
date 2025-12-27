"""Trace context management for distributed tracing across domains."""

from __future__ import annotations

import contextvars
import uuid
from typing import Optional

# Context variable for trace_id
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id",
    default=None,
)


class TraceContext:
    """
    Thread-safe and async-safe trace context manager.
    
    Provides automatic trace_id propagation across:
    - Async tasks
    - Event handlers
    - Domain boundaries
    
    Usage:
        from backend.core.trace_context import trace_context
        
        # Set trace_id for current context
        trace_context.set("abc-123-def")
        
        # Get current trace_id
        trace_id = trace_context.get()
        
        # Generate new trace_id
        trace_id = trace_context.generate()
        
        # Use as context manager
        with trace_context.scope("my-trace-id"):
            # All operations here have trace_id="my-trace-id"
            await some_async_function()
    """
    
    @staticmethod
    def get() -> Optional[str]:
        """
        Get current trace_id from context.
        
        Returns:
            Current trace_id or None if not set
        """
        return _trace_id_var.get()
    
    @staticmethod
    def set(trace_id: str) -> None:
        """
        Set trace_id for current context.
        
        Args:
            trace_id: Trace ID to set
        """
        _trace_id_var.set(trace_id)
    
    @staticmethod
    def generate() -> str:
        """
        Generate new trace_id and set it in context.
        
        Returns:
            Generated trace_id (UUID format)
        """
        trace_id = uuid.uuid4().hex
        _trace_id_var.set(trace_id)
        return trace_id
    
    @staticmethod
    def get_or_generate() -> str:
        """
        Get current trace_id or generate new one if not set.
        
        Returns:
            Trace ID
        """
        trace_id = _trace_id_var.get()
        if trace_id is None:
            trace_id = uuid.uuid4().hex
            _trace_id_var.set(trace_id)
        return trace_id
    
    @staticmethod
    def clear() -> None:
        """Clear trace_id from current context."""
        _trace_id_var.set(None)
    
    class scope:
        """
        Context manager for scoped trace_id.
        
        Automatically restores previous trace_id when exiting scope.
        """
        
        def __init__(self, trace_id: Optional[str] = None):
            """
            Create trace scope.
            
            Args:
                trace_id: Trace ID to use (or generate if None)
            """
            self.trace_id = trace_id or uuid.uuid4().hex
            self.previous_trace_id: Optional[str] = None
        
        def __enter__(self) -> str:
            """Enter scope and set trace_id."""
            self.previous_trace_id = _trace_id_var.get()
            _trace_id_var.set(self.trace_id)
            return self.trace_id
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            """Exit scope and restore previous trace_id."""
            _trace_id_var.set(self.previous_trace_id)
            return False


# Global instance
trace_context = TraceContext()
