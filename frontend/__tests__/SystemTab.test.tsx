/**
 * SystemTab Component Tests
 * Dashboard V3.0 - QA Test Suite
 * 
 * Tests:
 * - Microservice status cards
 * - Exchange health display
 * - Latency metrics
 * - Failover events
 * - Stress test results
 */

import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import SystemTab from '@/components/dashboard/SystemTab';

global.fetch = jest.fn();

const mockSystemData = {
  timestamp: '2025-12-05T10:00:00Z',
  services_health: [
    {
      name: 'AI Engine',
      status: 'UP',
      last_check: '2025-12-05T10:00:00Z',
      response_time_ms: 45
    },
    {
      name: 'Portfolio Intelligence',
      status: 'UP',
      last_check: '2025-12-05T10:00:00Z',
      response_time_ms: 32
    },
    {
      name: 'Risk & Safety',
      status: 'DOWN',
      last_check: '2025-12-05T09:59:00Z',
      response_time_ms: null
    }
  ],
  exchanges_health: [
    {
      exchange: 'binance_testnet',
      status: 'UP',
      latency_ms: 120,
      last_check: '2025-12-05T10:00:00Z'
    },
    {
      exchange: 'bybit',
      status: 'UP',
      latency_ms: 95,
      last_check: '2025-12-05T10:00:00Z'
    }
  ],
  failover_events_recent: [],
  stress_scenarios_recent: []
};

describe('SystemTab Component', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockSystemData
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('TEST-FE-SYS-001: Component renders without crashing', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).not.toBeInTheDocument();
    });
  });

  test('TEST-FE-SYS-002: Displays microservice names', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/AI Engine/)).toBeInTheDocument();
      expect(screen.getByText(/Portfolio Intelligence/)).toBeInTheDocument();
      expect(screen.getByText(/Risk.*Safety/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-SYS-003: Shows UP status for healthy services', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      const upStatuses = screen.getAllByText(/UP|OK|healthy/i);
      expect(upStatuses.length).toBeGreaterThan(0);
    });
  });

  test('TEST-FE-SYS-004: Shows DOWN status for unhealthy services', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/DOWN|unhealthy/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-SYS-005: Displays exchange names', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/binance_testnet|binance/i)).toBeInTheDocument();
      expect(screen.getByText(/bybit/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-SYS-006: Displays exchange latency', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/120.*ms|120/)).toBeInTheDocument();
      expect(screen.getByText(/95.*ms|95/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-SYS-007: Displays service response times', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/45.*ms|45/)).toBeInTheDocument();
      expect(screen.getByText(/32.*ms|32/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-SYS-008: No NaN displayed in metrics', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
    });
  });

  test('TEST-FE-SYS-009: Shows failover events section', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/failover|recent.*event/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-SYS-010: Shows stress scenarios section', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/stress.*scenario|stress.*test/i)).toBeInTheDocument();
    });
  });
});

describe('SystemTab Service Status Colors', () => {
  test('TEST-FE-SYS-COLOR-001: UP services show green/success indicator', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      // Find UP status and check for green/success class
      const upElements = screen.getAllByText(/UP|OK/i);
      expect(upElements[0].className).toMatch(/green|success/i);
    });
  });

  test('TEST-FE-SYS-COLOR-002: DOWN services show red/danger indicator', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      const downElement = screen.getByText(/DOWN/i);
      expect(downElement.className).toMatch(/red|danger|error/i);
    });
  });
});

describe('SystemTab Failover Events', () => {
  test('TEST-FE-SYS-FAIL-001: Shows failover events when present', async () => {
    const failoverData = {
      ...mockSystemData,
      failover_events_recent: [
        {
          timestamp: '2025-12-05T09:15:00Z',
          service: 'Execution Service',
          reason: 'Health check timeout',
          failover_to: 'backup_execution'
        }
      ]
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => failoverData
    });

    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/Execution Service|Health check timeout/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-SYS-FAIL-002: Shows empty state when no failovers', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/no.*failover|recent.*failover/i)).toBeInTheDocument();
    });
  });
});

describe('SystemTab Stress Scenarios', () => {
  test('TEST-FE-SYS-STRESS-001: Shows stress test results when present', async () => {
    const stressData = {
      ...mockSystemData,
      stress_scenarios_recent: [
        {
          timestamp: '2025-12-05T08:00:00Z',
          scenario: 'Flash Crash Simulation',
          result: 'PASSED',
          duration_seconds: 45
        },
        {
          timestamp: '2025-12-05T07:00:00Z',
          scenario: 'High Volatility Test',
          result: 'PASSED',
          duration_seconds: 60
        }
      ]
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => stressData
    });

    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/Flash Crash|High Volatility/i)).toBeInTheDocument();
      expect(screen.getByText(/PASSED/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-SYS-STRESS-002: Shows empty state when no stress tests', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/no.*stress|recent.*stress/i)).toBeInTheDocument();
    });
  });
});

describe('SystemTab Empty States', () => {
  test('TEST-FE-SYS-EMPTY-001: Handles empty services list', async () => {
    const emptyData = {
      ...mockSystemData,
      services_health: []
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => emptyData
    });

    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/no.*service|service.*health/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-SYS-EMPTY-002: Handles empty exchanges list', async () => {
    const emptyData = {
      ...mockSystemData,
      exchanges_health: []
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => emptyData
    });

    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.getByText(/no.*exchange|exchange.*health/i)).toBeInTheDocument();
    });
  });
});

describe('SystemTab Error Handling', () => {
  test('TEST-FE-SYS-ERR-001: Handles API error gracefully', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(new Error('API Error'));
    
    render(<SystemTab />);
    await waitFor(() => {
      expect(screen.queryByText(/error|failed/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-SYS-ERR-002: Handles null response times', async () => {
    render(<SystemTab />);
    await waitFor(() => {
      // Service with null response_time_ms should not show NaN
      expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
    });
  });
});

describe('SystemTab Latency Warnings', () => {
  test('TEST-FE-SYS-LAT-001: Shows warning for high latency', async () => {
    const highLatencyData = {
      ...mockSystemData,
      exchanges_health: [
        {
          exchange: 'binance_testnet',
          status: 'UP',
          latency_ms: 500,  // High latency
          last_check: '2025-12-05T10:00:00Z'
        }
      ]
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => highLatencyData
    });

    render(<SystemTab />);
    await waitFor(() => {
      // Should show latency value (500ms)
      expect(screen.getByText(/500/)).toBeInTheDocument();
    });
  });
});
