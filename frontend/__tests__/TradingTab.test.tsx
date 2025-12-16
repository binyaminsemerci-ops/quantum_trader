/**
 * TradingTab Component Tests
 * Dashboard V3.0 - QA Test Suite
 * 
 * Tests:
 * - Position table rendering
 * - Position data display
 * - Recent orders list
 * - Recent signals list
 * - Empty state handling
 */

import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import TradingTab from '@/components/dashboard/TradingTab';

global.fetch = jest.fn();

jest.mock('@/hooks/useDashboardStream', () => ({
  useDashboardStream: () => ({
    data: null,
    connected: false
  })
}));

const mockTradingData = {
  timestamp: '2025-12-05T10:00:00Z',
  open_positions: [
    {
      symbol: 'BTCUSDT',
      side: 'SELL',
      size: 0.001,
      entry_price: 95000.0,
      current_price: 96360.0,
      unrealized_pnl: -8.36,
      unrealized_pnl_pct: -1.43,
      value: 96.36,
      leverage: 20
    },
    {
      symbol: 'ETHUSDT',
      side: 'BUY',
      size: 0.1,
      entry_price: 3500.0,
      current_price: 3600.0,
      unrealized_pnl: 10.0,
      unrealized_pnl_pct: 2.86,
      value: 360.0,
      leverage: 10
    }
  ],
  recent_orders: [
    {
      id: 1,
      timestamp: '2025-12-05T09:30:00Z',
      account: 'default',
      exchange: 'binance_testnet',
      symbol: 'BTCUSDT',
      side: 'BUY',
      order_type: 'MARKET',
      size: 0.001,
      price: 95000.0,
      status: 'FILLED',
      strategy_id: 'quantum_trader_normal'
    },
    {
      id: 2,
      timestamp: '2025-12-05T09:45:00Z',
      account: 'default',
      exchange: 'binance_testnet',
      symbol: 'ETHUSDT',
      side: 'SELL',
      order_type: 'LIMIT',
      size: 0.1,
      price: 3500.0,
      status: 'NEW',
      strategy_id: 'ai_ensemble'
    }
  ],
  recent_signals: [
    {
      id: 'sig_1',
      timestamp: '2025-12-05T09:25:00Z',
      symbol: 'BTCUSDT',
      direction: 'LONG',
      confidence: 0.85,
      strategy_id: 'ai_ensemble'
    },
    {
      id: 'sig_2',
      timestamp: '2025-12-05T09:40:00Z',
      symbol: 'SOLUSDT',
      direction: 'SHORT',
      confidence: 0.72,
      strategy_id: 'quantum_trader_normal'
    }
  ],
  strategies_per_account: [
    {
      account: 'main',
      strategy_name: 'quantum_trader_normal',
      enabled: true,
      profile: 'normal',
      exchanges: ['binance_testnet'],
      symbols: [],
      description: 'Main NORMAL strategy (max 5 positions)',
      min_confidence: 0.65
    },
    {
      account: 'main',
      strategy_name: 'ai_ensemble',
      enabled: true,
      profile: 'normal',
      exchanges: ['binance_testnet'],
      symbols: [],
      description: '4-model AI ensemble (XGB+TFT+LSTM+RF)',
      min_confidence: 0.65
    }
  ]
};

describe('TradingTab Component', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockTradingData
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('TEST-FE-TR-001: Component renders without crashing', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.queryByText(/loading/i)).not.toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-002: Renders positions table', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      // Should show position symbols
      expect(screen.getByText(/BTCUSDT/)).toBeInTheDocument();
      expect(screen.getByText(/ETHUSDT/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-003: Displays position size', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      // Should show position sizes
      expect(screen.getByText(/0\.001|0\.1/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-004: Displays position side (BUY/SELL)', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/SELL|SHORT/i)).toBeInTheDocument();
      expect(screen.getByText(/BUY|LONG/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-005: Displays unrealized PnL', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      // Should show PnL values
      expect(screen.getByText(/-8\.36|10\.0/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-006: No NaN displayed in position table', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-007: Displays entry price', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/95000|3500/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-008: Displays current price', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/96360|3600/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-009: Displays leverage', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/20x|10x/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-010: Shows position count', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/2.*position|position.*2/i)).toBeInTheDocument();
    });
  });
});

describe('TradingTab Empty States', () => {
  test('TEST-FE-TR-EMPTY-001: Shows placeholder when no positions', async () => {
    const emptyData = {
      ...mockTradingData,
      open_positions: []
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => emptyData
    });

    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/no.*position|empty/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-EMPTY-002: Shows placeholder when no orders', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      // Should show "no orders" or similar
      expect(screen.getByText(/no.*order|recent.*order/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-EMPTY-003: Shows placeholder when no signals', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/no.*signal|recent.*signal/i)).toBeInTheDocument();
    });
  });
});

describe('TradingTab Error Handling', () => {
  test('TEST-FE-TR-ERR-001: Handles API error gracefully', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(new Error('API Error'));
    
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.queryByText(/error|failed/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-ERR-002: Handles malformed position data', async () => {
    const malformedData = {
      ...mockTradingData,
      open_positions: [
        {
          symbol: 'BTCUSDT',
          // Missing required fields
        }
      ]
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => malformedData
    });

    render(<TradingTab />);
    await waitFor(() => {
      // Should not crash, show error or fallback
      expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
    });
  });
});

describe('TradingTab PnL Colors', () => {
  test('TEST-FE-TR-COLOR-001: Negative PnL shows in red', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      // Find negative PnL (-8.36) and check for red/danger class
      const negPnl = screen.getByText(/-8\.36/);
      expect(negPnl.className).toMatch(/red|danger|negative/i);
    });
  });

  test('TEST-FE-TR-COLOR-002: Positive PnL shows in green', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      // Find positive PnL (10.0) and check for green/success class
      const posPnl = screen.getByText(/10\.0/);
      expect(posPnl.className).toMatch(/green|success|positive/i);
    });
  });
});

describe('TradingTab Recent Activity', () => {
  test('TEST-FE-TR-ACT-001: Recent orders section exists', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/recent.*order/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-ACT-002: Recent signals section exists', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/recent.*signal/i)).toBeInTheDocument();
    });
  });
});

describe('TradingTab Recent Orders Panel - STEP 9', () => {
  test('TEST-FE-TR-ORD-001: Renders recent orders table', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/recent.*order.*50/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-ORD-002: Displays order symbols', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      const orderSymbols = screen.getAllByText(/BTCUSDT|ETHUSDT/);
      expect(orderSymbols.length).toBeGreaterThan(0);
    });
  });

  test('TEST-FE-TR-ORD-003: Displays order sides (BUY/SELL)', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/BUY/)).toBeInTheDocument();
      expect(screen.getByText(/SELL/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-ORD-004: Displays order status (FILLED/NEW)', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/FILLED/)).toBeInTheDocument();
      expect(screen.getByText(/NEW/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-ORD-005: Displays order sizes', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      const orderSizes = screen.getAllByText(/0\.001|0\.1/);
      expect(orderSizes.length).toBeGreaterThan(0);
    });
  });

  test('TEST-FE-TR-ORD-006: Displays order prices', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/95000|3500/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-ORD-007: Displays order timestamps', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      // Should show formatted time (09:30, 09:45, etc.)
      expect(screen.getByText(/09:30|09:45/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-ORD-008: Shows "No orders" when empty', async () => {
    const emptyData = {
      ...mockTradingData,
      recent_orders: []
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => emptyData
    });

    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/no.*order/i)).toBeInTheDocument();
    });
  });
});

describe('TradingTab Recent Signals Panel - STEP 9', () => {
  test('TEST-FE-TR-SIG-001: Renders recent signals list', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/recent.*signal.*20/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-SIG-002: Displays signal symbols', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      const signalSymbols = screen.getAllByText(/BTCUSDT|SOLUSDT/);
      expect(signalSymbols.length).toBeGreaterThan(0);
    });
  });

  test('TEST-FE-TR-SIG-003: Displays signal directions (LONG/SHORT)', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/LONG/)).toBeInTheDocument();
      expect(screen.getByText(/SHORT/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-SIG-004: Displays signal confidence percentages', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/85%|72%/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-SIG-005: High confidence signals highlighted', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      // 85% confidence should have success/green styling
      const highConfidence = screen.getByText(/85%/);
      expect(highConfidence.className).toMatch(/green|success|high/i);
    });
  });

  test('TEST-FE-TR-SIG-006: LONG signals show green/up indicator', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      const longSignal = screen.getByText(/LONG/);
      expect(longSignal.className).toMatch(/green|success|up/i);
    });
  });

  test('TEST-FE-TR-SIG-007: SHORT signals show red/down indicator', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      const shortSignal = screen.getByText(/SHORT/);
      expect(shortSignal.className).toMatch(/red|danger|down/i);
    });
  });

  test('TEST-FE-TR-SIG-008: Shows "No signals" when empty', async () => {
    const emptyData = {
      ...mockTradingData,
      recent_signals: []
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => emptyData
    });

    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/no.*signal/i)).toBeInTheDocument();
    });
  });
});

describe('TradingTab Active Strategies Panel - STEP 9', () => {
  test('TEST-FE-TR-STR-001: Renders active strategies section', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/active.*strateg/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-STR-002: Displays strategy names', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/quantum_trader_normal/)).toBeInTheDocument();
      expect(screen.getByText(/ai_ensemble/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-STR-003: Displays strategy profiles', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      const profiles = screen.getAllByText(/normal/i);
      expect(profiles.length).toBeGreaterThan(0);
    });
  });

  test('TEST-FE-TR-STR-004: Displays strategy descriptions', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/main.*normal.*strategy/i)).toBeInTheDocument();
      expect(screen.getByText(/4-model.*ai.*ensemble/i)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-STR-005: Displays min confidence thresholds', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      const confidences = screen.getAllByText(/65%|0\.65/);
      expect(confidences.length).toBeGreaterThan(0);
    });
  });

  test('TEST-FE-TR-STR-006: Shows enabled status', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      // Both strategies are enabled
      const enabledIndicators = screen.getAllByText(/enabled|active|✓/i);
      expect(enabledIndicators.length).toBeGreaterThan(0);
    });
  });

  test('TEST-FE-TR-STR-007: Displays exchange info', async () => {
    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/binance_testnet/)).toBeInTheDocument();
    });
  });

  test('TEST-FE-TR-STR-008: Shows "No strategies" when empty', async () => {
    const emptyData = {
      ...mockTradingData,
      strategies_per_account: []
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => emptyData
    });

    render(<TradingTab />);
    await waitFor(() => {
      expect(screen.getByText(/no.*strateg/i)).toBeInTheDocument();
    });
  });
});

describe('TradingTab Data Polling - STEP 9', () => {
  jest.useFakeTimers();

  test('TEST-FE-TR-POLL-001: Polls data every 3 seconds', async () => {
    render(<TradingTab />);
    
    expect(global.fetch).toHaveBeenCalledTimes(1);
    
    // Fast-forward 3 seconds
    jest.advanceTimersByTime(3000);
    
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });
  });

  test('TEST-FE-TR-POLL-002: Updates data on poll', async () => {
    const { rerender } = render(<TradingTab />);
    
    await waitFor(() => {
      expect(screen.getByText(/BTCUSDT/)).toBeInTheDocument();
    });
    
    // Update mock data
    const updatedData = {
      ...mockTradingData,
      recent_orders: [
        ...mockTradingData.recent_orders,
        {
          id: 3,
          timestamp: '2025-12-05T10:00:00Z',
          account: 'default',
          exchange: 'binance_testnet',
          symbol: 'SOLUSDT',
          side: 'BUY',
          order_type: 'MARKET',
          size: 1.0,
          price: 150.0,
          status: 'FILLED',
          strategy_id: 'ai_ensemble'
        }
      ]
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => updatedData
    });
    
    jest.advanceTimersByTime(3000);
    
    await waitFor(() => {
      expect(screen.getByText(/SOLUSDT/)).toBeInTheDocument();
    });
  });

  afterAll(() => {
    jest.useRealTimers();
  });
});
