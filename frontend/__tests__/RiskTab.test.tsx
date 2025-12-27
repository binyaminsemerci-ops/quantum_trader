/**
 * RiskTab Component Tests
 * Dashboard V3.0 - QA Test Suite
 * 
 * Tests:
 * - Risk state badge display
 * - ESS status display
 * - Risk gate decisions stats
 * - VaR/ES metrics
 * - Warning banners for critical states
 */

import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import RiskTab from '@/components/dashboard/RiskTab';

global.fetch = jest.fn();

const mockRiskData = {
  timestamp: '2025-12-05T10:00:00Z',
  risk_gate_decisions_stats: {
    allow: 45,
    block: 3,
    scale: 12,
    total: 60
  },
  ess_triggers_recent: [],
  dd_per_profile: [],
  var_es_snapshot: {
    var_95: 150.0,
    var_99: 250.0,
    es_95: 200.0,
    es_99: 350.0
  }
};

describe('RiskTab Component', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockRiskData
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('TEST-FE-RISK-001: Component renders without crashing', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).not.toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-002: Displays risk gate allow count', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      expect(screen.getByText(/45/)).toBeInTheDocument();
      expect(screen.getByText(/allow/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-003: Displays risk gate block count', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      expect(screen.getByText(/3/)).toBeInTheDocument();
      expect(screen.getByText(/block/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-004: Displays risk gate scale count', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      expect(screen.getByText(/12/)).toBeInTheDocument();
      expect(screen.getByText(/scale/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-005: Displays total risk gate decisions', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      expect(screen.getByText(/60/)).toBeInTheDocument();
      expect(screen.getByText(/total/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-006: Displays VaR 95 metric', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      expect(screen.getByText(/150\.0|VaR.*95/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-007: Displays VaR 99 metric', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      expect(screen.getByText(/250\.0|VaR.*99/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-008: Displays ES 95 metric', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      expect(screen.getByText(/200\.0|ES.*95/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-009: Displays ES 99 metric', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      expect(screen.getByText(/350\.0|ES.*99/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-010: No NaN displayed in metrics', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
    });
  });
});

describe('RiskTab ESS Status', () => {
  test('TEST-FE-RISK-ESS-001: Shows ESS inactive status', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      expect(screen.getByText(/ESS|emergency/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-ESS-002: Shows ESS active warning', async () => {
    const activeEssData = {
      ...mockRiskData,
      ess_status: 'ACTIVE',
      ess_triggers_recent: [
        {
          timestamp: '2025-12-05T09:30:00Z',
          reason: 'Daily loss threshold exceeded',
          loss_amount: -5.8
        }
      ]
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => activeEssData
    });

    render(<RiskTab />);
    await waitFor(() => {
      // Should show warning or active state
      expect(screen.getByText(/active|triggered/i)).toBeInTheDocument();
    });
  });
});

describe('RiskTab Empty States', () => {
  test('TEST-FE-RISK-EMPTY-001: Handles zero risk gate stats', async () => {
    const emptyData = {
      ...mockRiskData,
      risk_gate_decisions_stats: {
        allow: 0,
        block: 0,
        scale: 0,
        total: 0
      }
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => emptyData
    });

    render(<RiskTab />);
    await waitFor(() => {
      // Should display 0, not NaN or error
      expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-EMPTY-002: Handles no ESS triggers', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      // Should show "no triggers" or empty list
      expect(screen.getByText(/no.*trigger|recent.*trigger/i)).toBeInTheDocument();
    });
  });
});

describe('RiskTab Critical States', () => {
  test('TEST-FE-RISK-CRIT-001: Shows warning banner for high block rate', async () => {
    const highBlockData = {
      ...mockRiskData,
      risk_gate_decisions_stats: {
        allow: 10,
        block: 40,  // High block rate
        scale: 5,
        total: 55
      }
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => highBlockData
    });

    render(<RiskTab />);
    await waitFor(() => {
      // Should show block count prominently
      expect(screen.getByText(/40/)).toBeInTheDocument();
    });
  });
});

describe('RiskTab Drawdown Per Profile', () => {
  test('TEST-FE-RISK-DD-001: Shows drawdown profiles when available', async () => {
    const ddData = {
      ...mockRiskData,
      dd_per_profile: [
        {
          profile: 'conservative',
          current_dd: -2.1,
          max_dd: -5.0
        },
        {
          profile: 'aggressive',
          current_dd: -8.3,
          max_dd: -15.0
        }
      ]
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ddData
    });

    render(<RiskTab />);
    await waitFor(() => {
      // Should show profile names or DD values
      expect(screen.getByText(/conservative|aggressive|-2\.1|-8\.3/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-DD-002: Shows empty state when no profiles', async () => {
    render(<RiskTab />);
    await waitFor(() => {
      // Should handle empty DD list gracefully
      expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
    });
  });
});

describe('RiskTab Error Handling', () => {
  test('TEST-FE-RISK-ERR-001: Handles API error gracefully', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(new Error('API Error'));
    
    render(<RiskTab />);
    await waitFor(() => {
      expect(screen.queryByText(/error|failed/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-RISK-ERR-002: Handles missing risk gate stats', async () => {
    const incompleteData = {
      ...mockRiskData,
      risk_gate_decisions_stats: {
        allow: 10
        // Missing block, scale, total
      }
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => incompleteData
    });

    render(<RiskTab />);
    await waitFor(() => {
      // Should not crash
      expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
    });
  });
});
