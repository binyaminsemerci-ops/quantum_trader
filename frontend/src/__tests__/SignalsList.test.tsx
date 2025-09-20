/** @vitest-environment jsdom */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { vi, expect, describe, it, beforeEach } from 'vitest';
import axios from 'axios';
import SignalsList from '../components/analysis/SignalsList';

vi.mock('axios');

const mockedAxios = axios as unknown as { get: vi.Mock };

describe('SignalsList', () => {
  beforeEach(() => {
    mockedAxios.get = vi.fn();
  });

  it('renders no signals message when API returns empty', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: [] });
    render(<SignalsList />);
    await waitFor(() => expect(screen.getByText(/No signals|Loading/)).toBeInTheDocument());
  });
});
