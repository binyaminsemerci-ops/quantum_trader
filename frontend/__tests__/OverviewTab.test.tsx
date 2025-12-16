/**
 * OverviewTab Component Tests
 * Dashboard V3.0 - QA Test Suite
 * 
 * Tests:
 * - Component rendering
 * - Environment badge display
 * - GO-LIVE status indicator
 * - PnL metrics display
 * - No NaN values in UI
 */

import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import OverviewTab from '@/components/dashboard/OverviewTab';

// Mock fetch
global.fetch = jest.fn();

// Mock WebSocket hook
jest.mock('@/hooks/useDashboardStream', () => ({
  useDashboardStream: () => ({
    data: null,
    connected: false
  })
}));

const mockOverviewData = {
  timestamp: '2025-12-05T10:00:00Z',
  environment: 'TESTNET',
  go_live_active: false,
  global_pnl: {
    equity: 816.61,
    cash: 892.94,
    daily_pnl: -76.33,
    daily_pnl_pct: -9.35,
    weekly_pnl: 0.0,
    monthly_pnl: 0.0,
    total_pnl: -76.33
  },
  positions_count: 10,
  exposure_per_exchange: [
    { exchange: 'binance_testnet', exposure: 41498.5 }
  ],
  risk_state: 'OK',
  ess_status: {
    status: 'INACTIVE',
    triggers_today: 0,
    daily_loss: 0.0,
    threshold: -5.0
  },
  capital_profiles_summary: []
};

describe('OverviewTab Component', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockOverviewData
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('TEST-FE-OV-001: Component renders without crashing', async () => {
    render(<OverviewTab />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).not.toBeInTheDocument();
    });
  });

  test('TEST-FE-OV-002: Displays environment badge (TESTNET)', async () => {
    render(<OverviewTab />);
    await waitFor(() => {
      expect(screen.getByText(/TESTNET/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-OV-003: Displays GO-LIVE status (INACTIVE)', async () => {
    render(<OverviewTab />);
    await waitFor(() => {
      // Look for inactive indicator
      const goLiveText = screen.queryByText(/GO-LIVE|inactive|live/i);
      expect(goLiveText).toBeInTheDocument();
    });
  });

  test('TEST-FE-OV-004: Displays equity value', async () => {
    render(<OverviewTab />);
    await waitFor(() => {
      // Should show equity value (816.61)
      expect(screen.getByText(/816\.61|equity/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-OV-005: Displays daily PnL', async () => {
    render(<OverviewTab />);
    await waitFor(() => {
      // Should show daily PnL (-76.33)
      expect(screen.getByText(/-76\.33|daily.*pnl/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-OV-006: No NaN displayed in UI', async () => {
    render(<OverviewTab />);
    await waitFor(() => {
      // Verify no "NaN" text appears
      expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
    });
  });

  test('TEST-FE-OV-007: Displays positions count', async () => {
    render(<OverviewTab />);
    await waitFor(() => {
      expect(screen.getByText(/10.*position|position.*10/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-OV-008: Displays risk state (OK)', async () => {
    render(<OverviewTab />);
    await waitFor(() => {
      expect(screen.getByText(/OK|risk.*ok/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-OV-009: Displays ESS status (INACTIVE)', async () => {
    render(<OverviewTab />);
    await waitFor(() => {
      expect(screen.getByText(/ESS.*INACTIVE|inactive.*ESS/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-OV-010: Handles loading state', () => {
    (global.fetch as jest.Mock).mockImplementation(() => new Promise(() => {}));
    render(<OverviewTab />);
    // Should show loading skeleton
    expect(document.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  test('TEST-FE-OV-011: Handles API error gracefully', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(new Error('API Error'));
    render(<OverviewTab />);
    await waitFor(() => {
      // Should show error message or fallback UI
      expect(screen.queryByText(/error|failed/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-OV-012: Shows exposure per exchange', async () => {
    render(<OverviewTab />);
    await waitFor(() => {
      expect(screen.getByText(/binance_testnet|exposure/i)).toBeInTheDocument();
    });
  });
});

describe('OverviewTab Empty States', () => {
  test('TEST-FE-OV-EMPTY-001: Handles zero equity', async () => {
    const emptyData = {
      ...mockOverviewData,
      global_pnl: { ...mockOverviewData.global_pnl, equity: 0 }
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => emptyData
    });

    render(<OverviewTab />);
    await waitFor(() => {
      // Should display 0.00, not NaN
      expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
    });
  });

  test('TEST-FE-OV-EMPTY-002: Handles zero positions', async () => {
    const emptyData = {
      ...mockOverviewData,
      positions_count: 0
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => emptyData
    });

    render(<OverviewTab />);
    await waitFor(() => {
      expect(screen.getByText(/0.*position|no.*position/i)).toBeInTheDocument();
    });
  });
});

describe('OverviewTab Critical States', () => {
  test('TEST-FE-OV-CRIT-001: Shows warning for CRITICAL risk state', async () => {
    const criticalData = {
      ...mockOverviewData,
      risk_state: 'CRITICAL'
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => criticalData
    });

    render(<OverviewTab />);
    await waitFor(() => {
      expect(screen.getByText(/CRITICAL|danger/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-OV-CRIT-002: Shows warning for ACTIVE ESS', async () => {
    const activeEssData = {
      ...mockOverviewData,
      ess_status: {
        ...mockOverviewData.ess_status,
        status: 'ACTIVE',
        triggers_today: 1
      }
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => activeEssData
    });

    render(<OverviewTab />);
    await waitFor(() => {
      expect(screen.getByText(/ACTIVE|triggered/i)).toBeInTheDocument();
    });
  });
});
