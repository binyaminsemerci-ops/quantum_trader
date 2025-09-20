/** @vitest-environment jsdom */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { expect, describe, it, beforeEach } from 'vitest';
import { vi } from 'vitest';
import * as matchers from '@testing-library/jest-dom/matchers';

expect.extend(matchers);
import TradeHistory from '../components/dashboard/TradeHistory';
import * as apiModule from '../utils/api';

vi.mock('../utils/api');
// avoid using vitest-specific types at top-level; treat mockedApi as any for test-time mocking
const mockedApi: any = apiModule;

describe('TradeHistory', () => {
  beforeEach(() => {
    mockedApi.api = { getTrades: vi.fn() } as any;
  });

  it('shows empty state when API returns no trades', async () => {
    mockedApi.api.getTrades.mockResolvedValueOnce({ data: [] });
    render(<TradeHistory />);
    await waitFor(() => expect(screen.getByText(/No trades found/)).toBeInTheDocument());
  });
});
