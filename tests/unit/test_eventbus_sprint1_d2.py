"""
SPRINT 1 - D2: EventBus Streams + Disk Buffer Tests

Tests for modular EventBus components:
- DiskBuffer: Local persistence during Redis outages (NO Redis required)
- RedisStreamBus: Redis Streams wrapper (requires Redis)
- EventBus: Integrated publish/subscribe with fallback (requires Redis)

Note: Redis-dependent tests are automatically skipped if Redis is unavailable.
"""

import asyncio
import json
import pytest
import redis.asyncio as redis
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from backend.core.eventbus import DiskBuffer, RedisStreamBus
from backend.core.event_bus import EventBus


# ==============================================================================
# Redis Availability Helper
# ==============================================================================

def is_redis_available() -> bool:
    """
    Check if Redis is available on localhost:6379.
    
    Returns:
        True if Redis responds to PING, False otherwise.
    """
    import socket
    try:
        # Quick socket check first (faster than full Redis connection)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex(('localhost', 6379))
        sock.close()
        
        if result != 0:
            return False
        
        # If socket connects, verify Redis responds
        import redis as sync_redis
        client = sync_redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


# Skip marker for Redis-dependent tests
REDIS_AVAILABLE = is_redis_available()
requires_redis = pytest.mark.skipif(
    not REDIS_AVAILABLE,
    reason="Redis not available on localhost:6379"
)


# ==============================================================================
# Test DiskBuffer
# ==============================================================================

class TestDiskBuffer:
    """Test disk buffer for event persistence."""
    
    @pytest.fixture
    def buffer(self, tmp_path):
        """Create DiskBuffer in temp directory."""
        buffer_dir = tmp_path / "eventbus_buffer"
        return DiskBuffer(str(buffer_dir))
    
    def test_buffer_initialization(self, buffer, tmp_path):
        """Test buffer directory creation."""
        assert buffer.buffer_dir.exists()
        assert buffer.buffer_dir.is_dir()
    
    def test_write_event(self, buffer):
        """Test writing event to disk buffer."""
        event_type = "test.event"
        message = {
            "payload": json.dumps({"symbol": "BTCUSDT"}),
            "timestamp": "2025-12-04T10:00:00Z"
        }
        
        success = buffer.write(event_type, message)
        
        assert success is True
        assert buffer._current_file.exists()
        
        # Read file content
        with open(buffer._current_file) as f:
            lines = f.readlines()
        
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event_type"] == event_type
        assert entry["message"] == message
        assert "buffered_at" in entry
    
    def test_read_all_events(self, buffer):
        """Test reading all buffered events."""
        # Write multiple events
        events = [
            ("event1", {"data": "test1"}),
            ("event2", {"data": "test2"}),
            ("event3", {"data": "test3"}),
        ]
        
        for event_type, message in events:
            buffer.write(event_type, message)
        
        # Read all events
        buffered = buffer.read_all()
        
        assert len(buffered) == 3
        assert all("event_type" in e for e in buffered)
        assert all("message" in e for e in buffered)
        assert all("buffered_at" in e for e in buffered)
    
    def test_read_all_ordered_by_timestamp(self, buffer):
        """Test events are returned in chronological order."""
        import time
        
        # Write events with small delay
        for i in range(3):
            buffer.write(f"event{i}", {"index": i})
            time.sleep(0.01)  # Ensure different timestamps
        
        buffered = buffer.read_all()
        
        # Verify chronological order
        timestamps = [e["buffered_at"] for e in buffered]
        assert timestamps == sorted(timestamps)
    
    def test_clear_buffer(self, buffer):
        """Test clearing buffer files."""
        # Write events
        buffer.write("test.event", {"data": "test"})
        current_file = buffer._current_file
        assert current_file.exists()
        
        # Clear buffer
        deleted = buffer.clear()
        
        assert deleted == 1
        assert not current_file.exists()
        assert buffer._current_file is None
    
    def test_get_stats(self, buffer):
        """Test buffer statistics."""
        # Write events
        for i in range(5):
            buffer.write(f"event{i}", {"data": f"test{i}"})
        
        stats = buffer.get_stats()
        
        assert stats["file_count"] == 1
        assert stats["total_events"] == 5
        assert stats["oldest_event"] is not None
        assert "buffer_dir" in stats


# ==============================================================================
# Test RedisStreamBus (Requires Redis)
# ==============================================================================

@requires_redis
class TestRedisStreamBus:
    """Test Redis Streams wrapper. Requires Redis on localhost:6379."""
    
    @pytest.fixture
    async def redis_client(self):
        """Create test Redis client."""
        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        
        # Cleanup test streams
        try:
            keys = await client.keys("quantum:stream:test.*")
            if keys:
                await client.delete(*keys)
        except:
            pass
        
        yield client
        
        # Cleanup after test
        try:
            keys = await client.keys("quantum:stream:test.*")
            if keys:
                await client.delete(*keys)
        except:
            pass
        
        await client.aclose()
    
    @pytest.fixture
    async def stream_bus(self, redis_client):
        """Create RedisStreamBus."""
        return RedisStreamBus(redis_client, service_name="test_service")
    
    @pytest.mark.asyncio
    async def test_publish_event(self, stream_bus):
        """Test publishing event to Redis Stream."""
        event_type = "test.signal"
        payload = {"symbol": "BTCUSDT", "price": 50000}
        
        message_id = await stream_bus.publish(event_type, payload)
        
        assert message_id is not None
        assert isinstance(message_id, str)
    
    @pytest.mark.asyncio
    async def test_ensure_consumer_group(self, stream_bus):
        """Test consumer group creation."""
        event_type = "test.group"
        
        success = await stream_bus.ensure_consumer_group(event_type)
        
        assert success is True
        
        # Creating again should also succeed (idempotent)
        success = await stream_bus.ensure_consumer_group(event_type)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_read_messages(self, stream_bus):
        """Test reading messages from stream."""
        event_type = "test.read"
        
        # Ensure consumer group
        await stream_bus.ensure_consumer_group(event_type)
        
        # Publish test event
        payload = {"test": "data"}
        await stream_bus.publish(event_type, payload)
        
        # Read messages
        messages = await stream_bus.read_messages(event_type, count=1, block=1000)
        
        assert len(messages) >= 1
        message_id, data = messages[0]
        assert "payload" in data
    
    @pytest.mark.asyncio
    async def test_acknowledge_message(self, stream_bus):
        """Test message acknowledgment."""
        event_type = "test.ack"
        
        await stream_bus.ensure_consumer_group(event_type)
        await stream_bus.publish(event_type, {"test": "ack"})
        
        messages = await stream_bus.read_messages(event_type, count=1, block=1000)
        assert len(messages) >= 1
        
        message_id, _ = messages[0]
        success = await stream_bus.acknowledge(event_type, message_id)
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, stream_bus):
        """Test Redis health check."""
        is_healthy = await stream_bus.health_check()
        
        assert is_healthy is True
        assert stream_bus._is_healthy is True


# ==============================================================================
# Test EventBus Integration (Requires Redis)
# ==============================================================================

@requires_redis
class TestEventBusIntegration:
    """Test EventBus with disk buffer fallback. Requires Redis on localhost:6379."""
    
    @pytest.fixture
    async def redis_client(self):
        """Create test Redis client."""
        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        await client.flushdb()
        
        yield client
        await client.aclose()
    
    @pytest.fixture
    async def event_bus(self, redis_client, tmp_path):
        """Create EventBus with temp disk buffer."""
        buffer_path = tmp_path / "eventbus_buffer"
        bus = EventBus(
            redis_client,
            service_name="test_bus",
            disk_buffer_path=str(buffer_path)
        )
        await bus.initialize()
        return bus
    
    @pytest.mark.asyncio
    async def test_publish_to_redis(self, event_bus):
        """Test publishing event when Redis is available."""
        event_type = "test.publish"
        payload = {"test": "data"}
        
        message_id = await event_bus.publish(event_type, payload)
        
        assert message_id is not None
        assert message_id != "buffered"
    
    @pytest.mark.asyncio
    async def test_publish_with_redis_down(self, event_bus):
        """Test publishing falls back to disk when Redis fails."""
        # Mock Redis publish to fail
        event_bus.redis_stream.publish = AsyncMock(return_value=None)
        
        event_type = "test.buffer"
        payload = {"test": "fallback"}
        
        message_id = await event_bus.publish(event_type, payload)
        
        assert message_id == "buffered"
        assert event_bus._redis_available is False
        
        # Verify event was written to disk buffer
        buffered = event_bus.disk_buffer.read_all()
        assert len(buffered) >= 1
    
    @pytest.mark.asyncio
    async def test_replay_after_redis_recovery(self, event_bus):
        """Test buffered events are replayed when Redis recovers."""
        # Step 1: Buffer some events (simulate Redis down)
        event_bus.redis_stream.publish = AsyncMock(return_value=None)
        event_bus._redis_available = False
        
        for i in range(3):
            await event_bus.publish(f"test.replay{i}", {"index": i})
        
        # Verify events are buffered
        buffered = event_bus.disk_buffer.read_all()
        assert len(buffered) == 3
        
        # Step 2: Restore Redis mock and trigger replay
        event_bus.redis_stream.publish = AsyncMock(
            side_effect=lambda *args, **kwargs: f"msg_{args[0]}"
        )
        event_bus._redis_available = True
        
        await event_bus._replay_buffered_events()
        
        # Verify buffer was cleared after successful replay
        buffered_after = event_bus.disk_buffer.read_all()
        assert len(buffered_after) == 0
    
    @pytest.mark.asyncio
    async def test_no_message_loss(self, event_bus):
        """Test at-least-once delivery guarantee."""
        published_events = []
        
        # Publish events
        for i in range(10):
            payload = {"index": i, "test": "no_loss"}
            message_id = await event_bus.publish(f"test.noloss", payload)
            published_events.append(message_id)
        
        # Verify all events got a message ID
        assert len(published_events) == 10
        assert all(mid is not None for mid in published_events)


# ==============================================================================
# Test EventBus Interface (No Redis Required - Uses Mocks)
# ==============================================================================

class TestEventBusInterface:
    """Test EventBus interface without requiring Redis (uses mocks)."""
    
    @pytest.fixture
    async def mock_redis_client(self):
        """Create mock Redis client."""
        mock = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        mock.xadd = AsyncMock(return_value=b"1234-0")
        mock.xgroup_create = AsyncMock(return_value=True)
        mock.xreadgroup = AsyncMock(return_value=[])
        mock.xack = AsyncMock(return_value=1)
        return mock
    
    @pytest.fixture
    def event_bus_with_mock(self, mock_redis_client, tmp_path):
        """Create EventBus with mock Redis."""
        buffer_path = tmp_path / "eventbus_buffer"
        return EventBus(
            mock_redis_client,
            service_name="test_mock_bus",
            disk_buffer_path=str(buffer_path)
        )
    
    @pytest.mark.asyncio
    async def test_eventbus_has_disk_buffer(self, event_bus_with_mock):
        """Test EventBus has DiskBuffer instance."""
        assert event_bus_with_mock.disk_buffer is not None
        assert isinstance(event_bus_with_mock.disk_buffer, DiskBuffer)
    
    @pytest.mark.asyncio
    async def test_eventbus_has_redis_stream(self, event_bus_with_mock):
        """Test EventBus has RedisStreamBus instance."""
        assert event_bus_with_mock.redis_stream is not None
        assert isinstance(event_bus_with_mock.redis_stream, RedisStreamBus)
    
    @pytest.mark.asyncio
    async def test_publish_interface(self, event_bus_with_mock):
        """Test publish method accepts correct parameters."""
        # This tests the interface without actually connecting to Redis
        event_type = "test.interface"
        payload = {"test": "data"}
        
        # Mock the redis_stream.publish to return a message_id
        event_bus_with_mock.redis_stream.publish = AsyncMock(return_value="mock-msg-123")
        
        message_id = await event_bus_with_mock.publish(event_type, payload)
        
        assert message_id == "mock-msg-123"
        event_bus_with_mock.redis_stream.publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_to_disk_on_redis_failure(self, event_bus_with_mock):
        """Test EventBus falls back to disk buffer when Redis fails."""
        # Mock Redis publish to fail
        event_bus_with_mock.redis_stream.publish = AsyncMock(return_value=None)
        
        event_type = "test.fallback"
        payload = {"test": "disk"}
        
        message_id = await event_bus_with_mock.publish(event_type, payload)
        
        # Should return "buffered"
        assert message_id == "buffered"
        
        # Verify event was written to disk
        buffered = event_bus_with_mock.disk_buffer.read_all()
        assert len(buffered) == 1
        assert buffered[0]["event_type"] == event_type


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
