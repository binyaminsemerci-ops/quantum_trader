import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import '@testing-library/jest-dom';
import axios from 'axios';
import TradeHistory from '../components/dashboard/TradeHistory';

vi.mock('axios');
const mockedAxios = axios as unknown as { get: vi.Mock };

describe('TradeHistory', () => {
  beforeEach(() => {
    mockedAxios.get = vi.fn();
  });

  it('shows empty state when API returns no trades', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: [] });
    render(<TradeHistory />);
    await waitFor(() => expect(screen.getByText(/No trades found/)).toBeInTheDocument());
  });
});
