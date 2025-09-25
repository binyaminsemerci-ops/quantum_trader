import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, fireEvent, screen } from '@testing-library/react';
import Settings from '../pages/Settings';

// Mock the api module
vi.mock('../utils/api', () => {
  return {
    default: {
      getSettings: vi.fn().mockResolvedValue({ data: {} }),
      saveSettings: vi.fn().mockResolvedValue({ data: {} }),
    },
    setDefaultExchange: vi.fn(),
  };
});

import api from '../utils/api';

describe('Settings page', () => {
  beforeEach(() => {
    (api.getSettings as any).mockResolvedValue({ data: {} });
    (api.saveSettings as any).mockResolvedValue({ data: {} });
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  it('renders and allows saving', async () => {
    render(<Settings />);
    // Wait for the Save button to appear after effect completes
    await screen.findByRole('button', { name: /save/i });

    // fill required fields for default exchange (binance)
  const keyInput = await screen.findByLabelText(/Binance API Key/i, { selector: 'input' });
  const secretInput = await screen.findByLabelText(/Binance API Secret/i, { selector: 'input' });
    fireEvent.change(keyInput, { target: { value: 'test-key' } });
    fireEvent.change(secretInput, { target: { value: 'test-secret' } });

    const saveButton = await screen.findByRole('button', { name: /save/i });
    expect(saveButton).toBeDefined();

    fireEvent.click(saveButton);

    expect(api.saveSettings).toHaveBeenCalled();
  });

  it('validates missing keys for selected default exchange', async () => {
    render(<Settings />);
    // Wait for the Save button to appear
    await screen.findByRole('button', { name: /save/i });

  const select = await screen.findByRole('combobox');
    // switch to coinbase
    fireEvent.change(select, { target: { value: 'coinbase' } });

    const saveButton = await screen.findByRole('button', { name: /save/i });
    fireEvent.click(saveButton);

    const warning = await screen.findByText(/please provide/i);
    expect(warning).toBeDefined();
    expect(api.saveSettings).not.toHaveBeenCalled();
  });
});
