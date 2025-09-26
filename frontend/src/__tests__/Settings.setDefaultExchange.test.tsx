import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, fireEvent, screen } from '@testing-library/react';
import Settings from '../pages/Settings';

vi.mock('../utils/api', () => {
  return {
    default: {
      getSettings: vi.fn().mockResolvedValue({ data: {} }),
      saveSettings: vi.fn().mockResolvedValue({ data: {} }),
    },
    setDefaultExchange: vi.fn(),
  };
});

import api, { setDefaultExchange } from '../utils/api';

describe('Settings default exchange wiring', () => {
  beforeEach(() => {
    (api.getSettings as any).mockResolvedValue({ data: {} });
    (api.saveSettings as any).mockResolvedValue({ data: {} });
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  it('calls setDefaultExchange when saving new default', async () => {
  render(<Settings />);
  // wait for Save button to appear (indicates initial effect completed)
  await screen.findByRole('button', { name: /save/i });

    const select = await screen.findByRole('combobox');
    fireEvent.change(select, { target: { value: 'coinbase' } });

    // fill coinbase keys so validation passes
  const key = await screen.findByLabelText(/Coinbase API Key/i, { selector: 'input' });
  const secret = await screen.findByLabelText(/Coinbase API Secret/i, { selector: 'input' });
    fireEvent.change(key, { target: { value: 'k' } });
    fireEvent.change(secret, { target: { value: 's' } });

    const saveButton = await screen.findByRole('button', { name: /save/i });
    fireEvent.click(saveButton);

    expect((setDefaultExchange as any)).toHaveBeenCalledWith('coinbase');
  });
});
